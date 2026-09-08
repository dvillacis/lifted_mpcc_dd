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
//   .npz  = a ZIP archive of those members, stored UNCOMPRESSED (method 0), so
//           no deflate implementation is needed — only CRC-32, which is 15
//           lines.  Member names must end in .npy; NumPy takes the stem as the
//           array's key.
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
   explicit Writer(const std::string& path) : out_(path, std::ios::binary) {}
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
         u32(0x02014b50); u16(20); u16(20); u16(0); u16(0); u16(0); u16(0);
         u32(m.crc); u32((uint32_t)m.size); u32((uint32_t)m.size);
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
   struct Member { std::string name; uint64_t offset, size; uint32_t crc; };

   static size_t count(const std::vector<size_t>& shape) {
      size_t n = 1;
      for (size_t s : shape) n *= s;
      return n;
   }

   // One .npy member, stored uncompressed inside the ZIP.
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

      // -- the ZIP local file header, then the member bytes --
      const std::string fname = name + ".npy";
      const uint32_t c = crc32(buf.data(), buf.size());
      Member m{fname, pos_, buf.size(), c};
      u32(0x04034b50); u16(20); u16(0); u16(0); u16(0); u16(0);
      u32(c); u32((uint32_t)buf.size()); u32((uint32_t)buf.size());
      u16((uint16_t)fname.size()); u16(0);
      raw(fname.data(), fname.size());
      raw((const char*)buf.data(), buf.size());
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
   bool closed_ = false;
};

}  // namespace npz

#endif  // NPZ_WRITER_HPP
