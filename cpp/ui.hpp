// ui.hpp — the command-line output of tv_dd and tv_learn, in Homebrew's style:
//
//   ==> Section                    a section (bold, blue arrow)
//   ✔ What finished                a stage that ended well (green tick)
//   Warning: what went wrong        a stage that stopped early or failed (yellow)
//   Error: why the run stops        to stderr (red)
//     key                value      the facts of a section, aligned
//
// The iteration tables and the machine-read lines (CSV,...) are printed as
// before, outside this file.  Colour: --color auto|always|never; auto colours
// a terminal unless NO_COLOR is set, and CLICOLOR_FORCE=1 forces it.  Under
// mpirun (or srun) auto is plain: every rank's stdout is a pseudo-terminal
// there, whatever mpirun's own output is, so --color always (or
// CLICOLOR_FORCE=1) asks for colour.  Only the caller's rank prints: call
// these on rank 0.
#pragma once

#include <unistd.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace ui {

inline bool& color_on() {
   static bool on = false;
   return on;
}

// mode: auto | always | never
inline void init(const std::string& mode) {
   const char* no = std::getenv("NO_COLOR");
   const char* force = std::getenv("CLICOLOR_FORCE");
   const bool forced = force && *force && std::strcmp(force, "0") != 0;
   // under mpirun a rank's stdout is a pseudo-terminal whatever mpirun's own output
   // is (a file, a pipe), so the terminal test says nothing there: plain, unless forced
   const bool launched = std::getenv("OMPI_COMM_WORLD_SIZE") || std::getenv("PMIX_RANK") ||
                         std::getenv("PMI_RANK") || std::getenv("SLURM_STEP_ID");
   if (mode == "always") color_on() = true;
   else if (mode == "never") color_on() = false;
   else color_on() = !(no && *no) && (forced || (!launched && isatty(fileno(stdout))));
}

inline const char* esc(const char* code) { return color_on() ? code : ""; }
inline const char* bold() { return esc("\033[1m"); }
inline const char* blue() { return esc("\033[1;34m"); }
inline const char* green() { return esc("\033[1;32m"); }
inline const char* yellow() { return esc("\033[4;33m"); }
inline const char* red() { return esc("\033[4;31m"); }
inline const char* dim() { return esc("\033[2m"); }
inline const char* reset() { return esc("\033[0m"); }

inline std::string vfmt(const char* fmt, va_list ap) {
   va_list a2;
   va_copy(a2, ap);
   const int n = std::vsnprintf(nullptr, 0, fmt, a2);
   va_end(a2);
   std::string s(n > 0 ? n : 0, '\0');
   if (n > 0) std::vsnprintf(&s[0], n + 1, fmt, ap);
   return s;
}

#define UI_FORMAT(f)                \
   va_list ap;                      \
   va_start(ap, f);                 \
   const std::string m = vfmt(f, ap); \
   va_end(ap)

// ==> text
inline void header(const char* fmt, ...) {
   UI_FORMAT(fmt);
   std::printf("%s==>%s %s%s%s\n", blue(), reset(), bold(), m.c_str(), reset());
   std::fflush(stdout);
}
// ✔ text
inline void ok(const char* fmt, ...) {
   UI_FORMAT(fmt);
   std::printf("%s✔%s %s\n", green(), reset(), m.c_str());
   std::fflush(stdout);
}
// Warning: text
inline void warn(const char* fmt, ...) {
   UI_FORMAT(fmt);
   std::printf("%sWarning%s: %s\n", yellow(), reset(), m.c_str());
   std::fflush(stdout);
}
// Error: text (stderr)
inline void error(const char* fmt, ...) {
   UI_FORMAT(fmt);
   std::fprintf(stderr, "%sError%s: %s\n", red(), reset(), m.c_str());
}
//   key                value
inline void kv(const char* key, const char* fmt, ...) {
   UI_FORMAT(fmt);
   std::printf("  %-20s %s\n", key, m.c_str());
}
//   text (a continuation line of the section)
inline void line(const char* fmt, ...) {
   UI_FORMAT(fmt);
   std::printf("  %s\n", m.c_str());
}
// a run's end: ✔ when ok, else Warning
inline void outcome(bool good, const char* fmt, ...) {
   UI_FORMAT(fmt);
   if (good) std::printf("%s✔%s %s\n", green(), reset(), m.c_str());
   else std::printf("%sWarning%s: %s\n", yellow(), reset(), m.c_str());
   std::fflush(stdout);
}

#undef UI_FORMAT

}  // namespace ui
