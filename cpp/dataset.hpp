// Dataset manager for the multi-sample (training-pair) driver: turn a FOLDER of
// images — Kodak, BSDS, whatever is on disk — into the list of (u_clean, f)
// pairs the bilevel upper level averages over.
//
// It is a thin, deliberate layer on top of image_io.hpp: listing, selection,
// per-pair seeding and provenance. The actual decode is image_io::load_image
// unchanged (center-crop → resize → normalize → f = clip(u_clean + σ·N(0,1),
// 0, 1)), so a one-image run of the dataset driver produces exactly the
// instance dd_solve_2d would produce from the same file.
//
// Three input routes, chosen by inspecting the path:
//   directory   → every image file in it, SORTED. readdir order is filesystem
//                 order and differs between machines and even between runs
//                 after a rewrite, so sorting is what makes "the first 8 images
//                 of kodak" mean the same thing on the laptop and on the HPC.
//   *.txt       → a MANIFEST: one image path per line, `#` comments, blank
//                 lines ignored, relative paths resolved against the manifest's
//                 own directory. Order is AUTHORED order and is NOT sorted —
//                 the point of a manifest is to pin a specific set and sequence.
//   anything    → a single image file (so --limit 1 on one PNG is the
//     else        degenerate case, and the S=1 equivalence gate against
//                 dd_solve_2d is expressible).
// The dumped-instance .txt written by dump_data_2d.py is NOT a manifest and is
// rejected with a message that says so — it carries one instance plus its owner
// map, which is meaningless for a dataset.
//
// REPRODUCIBILITY. The noise seed of a pair is base_seed + its index in the
// FULL sorted listing, never its position in the selected slice. So --skip and
// --limit re-select which pairs are used without re-randomizing the ones that
// stay: the 8-pair run and the 16-pair run share the noise realizations of
// their first 8 images, which is what makes a training-set-size sweep an actual
// controlled experiment.
#ifndef DATASET_HPP
#define DATASET_HPP

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <dirent.h>
#include <sys/stat.h>

// Brings in the vendored stb_image WITH its implementation, so this header (like
// image_io.hpp itself) may be included in only one translation unit per binary.
// Every driver here is a single TU, so that is free.
#include "image_io.hpp"

namespace dataset {

// One training (or validation) pair, plus the provenance needed to reproduce it.
struct Pair {
   std::string path;
   int index = 0;         // position in the FULL listing — this seeds the noise
   unsigned seed = 0;
   int src_w = 0, src_h = 0;
   std::vector<double> uclean, f;
   double psnr_noisy = 0.0;   // filled by the driver (driver::psnr)
};

// skip → stride → limit, applied to the listing in that order. limit 0 = all.
struct Select {
   int skip = 0, stride = 1, limit = 0;
};

inline std::string lower(std::string s) {
   for (char& c : s) c = (char)std::tolower((unsigned char)c);
   return s;
}

inline bool is_dir(const std::string& p) {
   struct stat st;
   return stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

// The stb_image format set. Anything else in the folder (README, .npy, a
// nested directory) is skipped silently — pointing the driver at a data folder
// that also holds notes should not be an error.
inline bool is_image_ext(const std::string& p) {
   static const char* exts[] = {".png",  ".jpg", ".jpeg", ".bmp", ".tga",
                                ".gif",  ".pgm", ".ppm",  ".psd", ".hdr",
                                ".pic",  ".pnm"};
   const std::string l = lower(p);
   for (const char* e : exts)
      if (image_io::ends_with(l, e)) return true;
   return false;
}

inline std::string dirname_of(const std::string& p) {
   const size_t k = p.find_last_of('/');
   return k == std::string::npos ? std::string(".") : p.substr(0, k);
}

inline std::string join_path(const std::string& dir, const std::string& name) {
   if (name.empty() || name[0] == '/') return name;       // already absolute
   if (dir.empty() || dir == ".") return name;
   return dir + (dir.back() == '/' ? "" : "/") + name;
}

inline std::string trim(const std::string& s) {
   size_t a = s.find_first_not_of(" \t\r\n");
   if (a == std::string::npos) return "";
   size_t b = s.find_last_not_of(" \t\r\n");
   return s.substr(a, b - a + 1);
}

// Read a manifest: one image path per line, `#` comments, blank lines skipped.
// The WHOLE trimmed line is the path, so filenames with spaces work.
inline std::vector<std::string> read_manifest(const std::string& fn) {
   std::ifstream in(fn);
   if (!in) throw std::runtime_error("cannot open manifest " + fn);
   const std::string base = dirname_of(fn);
   std::vector<std::string> out;
   std::string line;
   int lineno = 0;
   while (std::getline(in, line)) {
      ++lineno;
      const size_t h = line.find('#');
      if (h != std::string::npos) line = line.substr(0, h);
      const std::string p = trim(line);
      if (p.empty()) continue;
      // The dumped-instance format of dump_data_2d.py also ends in .txt, and
      // silently reading its header numbers as filenames would fail much later
      // with a confusing stb_image error. Its first token is a number, which
      // can never carry an image extension, so this catches it on line 1.
      if (!is_image_ext(p))
         throw std::runtime_error(
             fn + ":" + std::to_string(lineno) + ": '" + p +
             "' is not an image path. A .txt input is read as a MANIFEST (one "
             "image path per line); the dumped-instance .txt of dump_data_2d.py "
             "holds a single instance and is not a dataset — point --data at a "
             "folder of images instead.");
      out.push_back(join_path(base, p));
   }
   if (out.empty()) throw std::runtime_error("manifest " + fn + " lists no images");
   return out;
}

// Directory → sorted image list; *.txt → manifest (authored order); else the
// single file itself.
inline std::vector<std::string> list_images(const std::string& spec) {
   if (is_dir(spec)) {
      DIR* d = opendir(spec.c_str());
      if (!d) throw std::runtime_error("cannot open directory " + spec);
      std::vector<std::string> out;
      while (struct dirent* e = readdir(d)) {
         const std::string name = e->d_name;
         if (name.empty() || name[0] == '.') continue;   // ".", "..", dotfiles
         if (!is_image_ext(name)) continue;
         out.push_back(join_path(spec, name));
      }
      closedir(d);
      if (out.empty())
         throw std::runtime_error("no image files in directory " + spec);
      std::sort(out.begin(), out.end());       // see the header: determinism
      return out;
   }
   if (image_io::ends_with(lower(spec), ".txt")) return read_manifest(spec);
   if (!is_image_ext(spec))
      throw std::runtime_error(
          "--data " + spec +
          ": not a directory, not a .txt manifest, and not a recognized image "
          "extension");
   return {spec};
}

// Which listing entries a Select picks. Separated out so the driver can carve a
// validation slice from the entries AFTER the training slice without loading
// anything twice.
inline std::vector<int> pick(const std::vector<std::string>& files,
                             const Select& sel) {
   if (sel.stride < 1) throw std::runtime_error("--stride must be >= 1");
   if (sel.skip < 0) throw std::runtime_error("--skip must be >= 0");
   std::vector<int> idx;
   for (int i = sel.skip; i < (int)files.size(); i += sel.stride) {
      idx.push_back(i);
      if (sel.limit > 0 && (int)idx.size() == sel.limit) break;
   }
   return idx;
}

// Decode the picked entries into pairs. `base` supplies size/sigma/normalize/
// area_downsample; the per-pair seed is set here and overrides base.seed.
inline std::vector<Pair> load_pairs(const std::vector<std::string>& files,
                                    const std::vector<int>& idx,
                                    const image_io::Opts& base,
                                    unsigned base_seed, bool allow_upsample) {
   std::vector<Pair> out(idx.size());
   for (size_t s = 0; s < idx.size(); ++s) {
      Pair& pr = out[s];
      pr.index = idx[s];
      pr.path = files[idx[s]];
      pr.seed = base_seed + (unsigned)idx[s];

      // Header-only probe BEFORE the decode, so a too-small source is refused
      // without paying for it. This is the PNG-IHDR guard the N=512/N=1024
      // SLURM scripts carry as an inline Python heredoc, moved into the loader
      // where it applies to every format and every driver that uses it: an
      // upsampled "clean" image has no detail to recover, so the run measures
      // the interpolator rather than the denoiser, and nothing in the output
      // would say so.
      int w = 0, h = 0, ch = 0;
      if (!stbi_info(pr.path.c_str(), &w, &h, &ch))
         throw std::runtime_error("stb_image cannot read " + pr.path + " (" +
                                  stbi_failure_reason() + ")");
      pr.src_w = w;
      pr.src_h = h;
      if (std::min(w, h) < base.size && !allow_upsample)
         throw std::runtime_error(
             pr.path + " is " + std::to_string(w) + "x" + std::to_string(h) +
             ", smaller than --size " + std::to_string(base.size) +
             " — it would be UPSAMPLED. Use a smaller --size, a larger source, "
             "or --allow-upsample to do it deliberately.");

      image_io::Opts o = base;
      o.seed = pr.seed;
      int N = 0;
      image_io::load_image(pr.path, o, N, pr.uclean, pr.f);
      if (N != base.size)
         throw std::runtime_error("internal: image_io returned N=" +
                                  std::to_string(N) + " for " + pr.path);
   }
   return out;
}

// CSV provenance: exactly what was loaded, in load order. Re-running the same
// command with the same --seed must reproduce this file byte for byte.
inline void write_manifest(const std::string& fn, const std::vector<Pair>& pairs,
                           int N, double sigma, const char* role,
                           bool append = false) {
   std::ofstream out(fn, append ? std::ios::app : std::ios::trunc);
   if (!out) throw std::runtime_error("cannot write " + fn);
   if (!append)
      out << "role,slot,listing_index,path,src_w,src_h,N,sigma,seed,psnr_noisy_dB\n";
   for (size_t s = 0; s < pairs.size(); ++s) {
      const Pair& p = pairs[s];
      out << role << "," << s << "," << p.index << "," << p.path << ","
          << p.src_w << "," << p.src_h << "," << N << "," << sigma << ","
          << p.seed << "," << p.psnr_noisy << "\n";
   }
}

}  // namespace dataset

#endif  // DATASET_HPP
