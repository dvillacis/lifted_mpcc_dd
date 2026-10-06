// npz_writer.hpp — write NumPy .npz archives from C++ with NO dependencies.
//
// WHY THIS EXISTS.  Solutions used to be dumped as whitespace-separated text:
// a positional token stream whose meaning lived only in the reader's head, at
// ~24 bytes per double, losing bits to decimal rounding, and with no way to
// tell what a file contains without knowing the layout by heart.  .npz fixes
// all four: SELF-DESCRIBING (every array has a name, dtype and shape), EXACT
// (raw little-endian IEEE-754 — bit-identical round-trip), COMPACT (8 bytes
// per double), and READABLE EVERYWHERE (np.load in Python; MATLAB, Julia and
// R all have readers).
//
// Why not HDF5, the usual "professional" answer: it would put libhdf5 on the
// critical path of a project that deliberately builds against Eigen alone and
// treats even MA57 as optional.  .npz costs nothing — the format is a ZIP of
// .npy members, and both halves are simple enough to emit correctly here:
//
//   .npy  = "\x93NUMPY" + version + a padded Python-dict header giving descr /
//           fortran_order / shape, then the raw buffer.  The 10-byte preamble
//           plus header must be a multiple of 64 bytes (NumPy aligns the data).
//   .npz  = a ZIP archive of those members.  Members are DEFLATED (method 8,
//           the only compressed method NumPy reads) through zlib, which is a
//           system library on macOS and every Linux — `-lz`, nothing to
//           install and nothing vendored.  Define NPZ_NO_ZLIB to drop the
//           dependency and emit stored (method 0) members instead; the files
//           stay valid either way.  Member names must end in .npy; NumPy takes
//           the stem as the array's key.
//
//           ZIP wants a RAW deflate stream (no zlib header/trailer), hence
//           deflateInit2 with negative windowBits.  The CRC-32 is always of
//           the UNCOMPRESSED bytes.  A member that fails to shrink is written
//           stored, which is what every ZIP writer does and costs nothing to
//           support since both paths already exist.
//
// Scalars are written as 0-d arrays, so `int(d["N"])` and `float(d["sigma"])`
// work on the Python side without special cases.
//
// Usage:
//     npz::Writer w("sol.npz");
//     w.scalar_i("N", 32);
//     w.scalar_d("sigma", 0.1);
//     w.array_d("u", u.data(), {32, 32});      // row-major, C order
//     if (!w.close()) { /* write failed */ }
#ifndef NPZ_WRITER_HPP
#define NPZ_WRITER_HPP

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#ifndef NPZ_NO_ZLIB
#include <zlib.h>
#endif

namespace npz {

// Standard CRC-32 (reflected, polynomial 0xEDB88320) — the checksum ZIP wants.
inline uint32_t crc32(const uint8_t* p, size_t n, uint32_t crc = 0) {
   static uint32_t tbl[256];
   static bool init = false;
   if (!init) {
      for (uint32_t i = 0; i < 256; ++i) {
         uint32_t c = i;
         for (int k = 0; k < 8; ++k) c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
         tbl[i] = c;
      }
      init = true;
   }
   crc = ~crc;
   for (size_t i = 0; i < n; ++i) crc = tbl[(crc ^ p[i]) & 0xFF] ^ (crc >> 8);
   return ~crc;
}

class Writer {
public:
   // level: zlib compression level, 0-9 (default 6 — the usual size/time knee;
   // these arrays are smooth image data and compress well).  Ignored when
   // built with NPZ_NO_ZLIB.
   explicit Writer(const std::string& path, int level = 6)
       : out_(path, std::ios::binary), level_(level) {}
   bool ok() const { return (bool)out_; }

   // ---- the public array API -------------------------------------------
   // shape is C-order (row-major), as NumPy's default.  An empty shape gives
   // a 0-d array, which is how scalars are stored.
   void array_d(const std::string& name, const double* v,
                const std::vector<size_t>& shape) {
      put(name, "<f8", (const uint8_t*)v, count(shape) * 8, shape);
   }
   void array_i(const std::string& name, const int32_t* v,
                const std::vector<size_t>& shape) {
      put(name, "<i4", (const uint8_t*)v, count(shape) * 4, shape);
   }
   void scalar_d(const std::string& name, double v) { array_d(name, &v, {}); }
   void scalar_i(const std::string& name, int32_t v) { array_i(name, &v, {}); }

   // Finish the archive: central directory + end-of-central-directory record.
   // MUST be called (and its result checked) — the file is invalid without it.
   bool close() {
      if (closed_) return (bool)out_;
      closed_ = true;
      const uint64_t cd_off = pos_;
      for (const Member& m : members_) {
         u32(0x02014b50); u16(20); u16(20); u16(0); u16(m.method); u16(0); u16(0);
         u32(m.crc); u32((uint32_t)m.csize); u32((uint32_t)m.size);
         u16((uint16_t)m.name.size()); u16(0); u16(0); u16(0); u16(0); u32(0);
         u32((uint32_t)m.offset);
         raw(m.name.data(), m.name.size());
      }
      const uint64_t cd_size = pos_ - cd_off;
      u32(0x06054b50); u16(0); u16(0);
      u16((uint16_t)members_.size()); u16((uint16_t)members_.size());
      u32((uint32_t)cd_size); u32((uint32_t)cd_off); u16(0);
      out_.flush();
      return (bool)out_;
   }

private:
   struct Member {
      std::string name;
      uint64_t offset, size, csize;      // size = uncompressed, csize = stored
      uint32_t crc;
      uint16_t method;
   };

#ifndef NPZ_NO_ZLIB
   // Raw deflate (negative windowBits ⇒ no zlib header/trailer, which is what
   // ZIP expects).  False on any zlib error, and the caller then stores.
   static bool deflate_raw(const std::vector<uint8_t>& in,
                           std::vector<uint8_t>& out, int level) {
      z_stream zs{};
      if (deflateInit2(&zs, level, Z_DEFLATED, -MAX_WBITS, 8,
                       Z_DEFAULT_STRATEGY) != Z_OK)
         return false;
      out.resize(deflateBound(&zs, (uLong)in.size()));
      zs.next_in = const_cast<Bytef*>(in.data());
      zs.avail_in = (uInt)in.size();
      zs.next_out = out.data();
      zs.avail_out = (uInt)out.size();
      const int rc = deflate(&zs, Z_FINISH);
      const uLong produced = zs.total_out;
      deflateEnd(&zs);
      if (rc != Z_STREAM_END) return false;
      out.resize(produced);
      return true;
   }
#endif

   static size_t count(const std::vector<size_t>& shape) {
      size_t n = 1;
      for (size_t s : shape) n *= s;
      return n;
   }

   // One .npy member: build the .npy bytes, then add them to the ZIP —
   // deflated when that shrinks them, stored otherwise.
   void put(const std::string& name, const char* descr, const uint8_t* data,
            size_t nbytes, const std::vector<size_t>& shape) {
      // -- the .npy header: a Python dict literal, space-padded so that the
      // 10-byte preamble + header is a multiple of 64 and the payload lands
      // on a 64-byte boundary (NumPy's alignment contract).
      std::string dict = std::string("{'descr': '") + descr +
                         "', 'fortran_order': False, 'shape': (";
      for (size_t i = 0; i < shape.size(); ++i) {
         dict += std::to_string(shape[i]);
         if (shape.size() == 1 || i + 1 < shape.size()) dict += ", ";
      }
      dict += "), }";
      size_t total = 10 + dict.size() + 1;               // +1 for the newline
      const size_t pad = (64 - total % 64) % 64;
      dict.append(pad, ' ');
      dict += '\n';

      std::string npy = "\x93NUMPY";
      npy += (char)1; npy += (char)0;                    // format version 1.0
      const uint16_t hlen = (uint16_t)dict.size();
      npy += (char)(hlen & 0xFF); npy += (char)((hlen >> 8) & 0xFF);
      npy += dict;

      std::vector<uint8_t> buf(npy.begin(), npy.end());
      buf.insert(buf.end(), data, data + nbytes);

      // The CRC is of the uncompressed member, whatever we do next.
      const uint32_t c = crc32(buf.data(), buf.size());
      uint16_t method = 0;
      const std::vector<uint8_t>* payload = &buf;
      std::vector<uint8_t> comp;
#ifndef NPZ_NO_ZLIB
      if (level_ > 0 && deflate_raw(buf, comp, level_) && comp.size() < buf.size()) {
         method = 8;                       // ZIP_DEFLATED — what NumPy reads
         payload = &comp;
      }
#endif
      const std::string fname = name + ".npy";
      Member m{fname, pos_, buf.size(), payload->size(), c, method};
      u32(0x04034b50); u16(20); u16(0); u16(method); u16(0); u16(0);
      u32(c); u32((uint32_t)payload->size()); u32((uint32_t)buf.size());
      u16((uint16_t)fname.size()); u16(0);
      raw(fname.data(), fname.size());
      raw((const char*)payload->data(), payload->size());
      members_.push_back(m);
   }

   // ---- little-endian primitives ---------------------------------------
   void raw(const char* p, size_t n) { out_.write(p, (std::streamsize)n); pos_ += n; }
   void u16(uint16_t v) { char b[2] = {(char)(v & 0xFF), (char)(v >> 8)}; raw(b, 2); }
   void u32(uint32_t v) {
      char b[4] = {(char)(v & 0xFF), (char)((v >> 8) & 0xFF),
                   (char)((v >> 16) & 0xFF), (char)((v >> 24) & 0xFF)};
      raw(b, 4);
   }

   std::ofstream out_;
   std::vector<Member> members_;
   uint64_t pos_ = 0;
   int level_ = 6;
   bool closed_ = false;
};

}  // namespace npz

#endif  // NPZ_WRITER_HPP
