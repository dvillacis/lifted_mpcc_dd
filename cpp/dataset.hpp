// dataset.hpp — which images tv_learn trains on and evaluates on.
//
// A --train / --val argument is
//   a directory  → every image file in it, SORTED (readdir order differs between
//                  machines, so sorting is what makes "the first 8 images" mean
//                  the same thing on the laptop and on the cluster);
//   a .txt file  → a MANIFEST: one image path per line, `#` comments, blank lines
//                  skipped, relative paths resolved against the manifest's own
//                  directory; the authored order is kept;
//   else         → a single image file.
// skip → stride → limit select from the listing.  The noise seed of an image is
// base seed + its index in the FULL listing, so --limit 8 and --limit 16 share
// the noise of their first 8 images.  Ported from ../archive/cpp/dataset.hpp.
#pragma once

#include <algorithm>
#include <cctype>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <dirent.h>
#include <sys/stat.h>

namespace dataset {

struct Select {
   int skip = 0, stride = 1, limit = 0;   // limit 0: all
};

inline std::string lower(std::string s) {
   for (char& c : s) c = (char)std::tolower((unsigned char)c);
   return s;
}
inline bool ends_with(const std::string& s, const std::string& t) {
   return s.size() >= t.size() && s.compare(s.size() - t.size(), t.size(), t) == 0;
}
inline bool is_dir(const std::string& p) {
   struct stat st;
   return stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}
inline bool is_image_ext(const std::string& p) {
   static const char* exts[] = {".png", ".jpg", ".jpeg", ".bmp", ".tga", ".gif",
                                ".pgm", ".ppm", ".psd", ".hdr", ".pic", ".pnm"};
   const std::string l = lower(p);
   for (const char* e : exts)
      if (ends_with(l, e)) return true;
   return false;
}
inline std::string dirname_of(const std::string& p) {
   const size_t k = p.find_last_of('/');
   return k == std::string::npos ? std::string(".") : p.substr(0, k);
}
inline std::string join_path(const std::string& dir, const std::string& name) {
   if (name.empty() || name[0] == '/') return name;
   if (dir.empty() || dir == ".") return name;
   return dir + (dir.back() == '/' ? "" : "/") + name;
}
inline std::string trim(const std::string& s) {
   const size_t a = s.find_first_not_of(" \t\r\n");
   if (a == std::string::npos) return "";
   const size_t b = s.find_last_not_of(" \t\r\n");
   return s.substr(a, b - a + 1);
}
// the file name without directory and extension
inline std::string stem(const std::string& p) {
   const size_t k = p.find_last_of('/');
   std::string s = k == std::string::npos ? p : p.substr(k + 1);
   const size_t d = s.find_last_of('.');
   return d == std::string::npos ? s : s.substr(0, d);
}

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
      if (!is_image_ext(p))
         throw std::runtime_error(fn + ":" + std::to_string(lineno) + ": '" + p +
                                  "' is not an image path (a .txt is read as a manifest)");
      out.push_back(join_path(base, p));
   }
   if (out.empty()) throw std::runtime_error("manifest " + fn + " lists no images");
   return out;
}

inline std::vector<std::string> list_images(const std::string& spec) {
   if (is_dir(spec)) {
      DIR* d = opendir(spec.c_str());
      if (!d) throw std::runtime_error("cannot open directory " + spec);
      std::vector<std::string> out;
      while (struct dirent* e = readdir(d)) {
         const std::string name = e->d_name;
         if (name.empty() || name[0] == '.' || !is_image_ext(name)) continue;
         out.push_back(join_path(spec, name));
      }
      closedir(d);
      if (out.empty()) throw std::runtime_error("no image files in directory " + spec);
      std::sort(out.begin(), out.end());
      return out;
   }
   if (ends_with(lower(spec), ".txt")) return read_manifest(spec);
   if (!is_image_ext(spec))
      throw std::runtime_error(spec + ": not a directory, a .txt manifest, or an image file");
   return {spec};
}

// the listing indices a Select picks
inline std::vector<int> pick(const std::vector<std::string>& files, const Select& sel) {
   if (sel.stride < 1 || sel.skip < 0) throw std::runtime_error("need --skip >= 0, --stride >= 1");
   std::vector<int> idx;
   for (int i = sel.skip; i < (int)files.size(); i += sel.stride) {
      idx.push_back(i);
      if (sel.limit > 0 && (int)idx.size() == sel.limit) break;
   }
   return idx;
}

}  // namespace dataset
