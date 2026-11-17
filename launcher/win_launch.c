/* Itch desktop launcher. WinPython lives under runtime\. Missing
 * dependencies are installed into that interpreter, then the game runs.
 * Rebuild: zig cc -target x86_64-windows-gnu -O2 -Wl,--subsystem,windows -o block-game.exe launcher/win_launch.c
 */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <shellapi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>
#include <wctype.h>

#define PATH_CAP 4096
#define DEPTH_MAX 6

static void fail(const wchar_t *msg) {
    MessageBoxW(NULL, msg, L"Block Game", MB_OK | MB_ICONERROR);
    ExitProcess(1);
}

static wchar_t *quote_arg(const wchar_t *s) {
    size_t n = wcslen(s);
    wchar_t *out = (wchar_t *)malloc((n * 2 + 3) * sizeof(wchar_t));
    wchar_t *p = out;
    if (!out) fail(L"Out of memory.");
    *p++ = L'"';
    for (; *s; ++s) {
        if (*s == L'"') *p++ = L'\\';
        *p++ = *s;
    }
    *p++ = L'"';
    *p = 0;
    return out;
}

static void append(wchar_t *dest, size_t cap, const wchar_t *src) {
    size_t used = wcslen(dest);
    size_t add = wcslen(src);
    if (used + add + 1 >= cap) fail(L"The launch command is too long.");
    memcpy(dest + used, src, (add + 1) * sizeof(wchar_t));
}

static int run_process(const wchar_t *cmdline, const wchar_t *cwd, DWORD flags,
                       HANDLE out, int *exit_code) {
    wchar_t *mutable = _wcsdup(cmdline);
    HANDLE nul = NULL;
    STARTUPINFOW si;
    PROCESS_INFORMATION pi;
    if (!mutable) fail(L"Out of memory.");
    ZeroMemory(&si, sizeof si);
    ZeroMemory(&pi, sizeof pi);
    si.cb = sizeof si;
    if (out) {
        SECURITY_ATTRIBUTES sa;
        sa.nLength = sizeof sa;
        sa.lpSecurityDescriptor = NULL;
        sa.bInheritHandle = TRUE;
        nul = CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, OPEN_EXISTING, 0, NULL);
        si.dwFlags = STARTF_USESTDHANDLES;
        si.hStdInput = nul;
        si.hStdOutput = out;
        si.hStdError = out;
    }
    if (!CreateProcessW(NULL, mutable, NULL, NULL, out != NULL, flags, NULL, cwd, &si, &pi)) {
        if (nul && nul != INVALID_HANDLE_VALUE) CloseHandle(nul);
        free(mutable);
        return 0;
    }
    if (nul && nul != INVALID_HANDLE_VALUE) CloseHandle(nul);
    free(mutable);
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    if (exit_code) *exit_code = (int)code;
    return 1;
}

static void lower_copy(wchar_t *dest, const wchar_t *src, size_t cap) {
    size_t i = 0;
    for (; src[i] && i + 1 < cap; ++i) dest[i] = towlower(src[i]);
    dest[i] = 0;
}

static int python_score(const wchar_t *path, const wchar_t *name, int depth) {
    wchar_t lower[PATH_CAP];
    lower_copy(lower, path, PATH_CAP);
    if (wcsstr(lower, L"\\.venv\\")) return -1;
    if (wcsstr(lower, L"\\scripts\\")) return -1;
    if (wcsstr(lower, L"\\lib\\")) return -1;
    if (wcsstr(lower, L"\\site-packages\\")) return -1;
    int score = 40 - depth;
    if (wcsstr(lower, L"winpython") || wcsstr(lower, L"wpy")) score += 100;
    if (_wcsicmp(name, L"python.exe") == 0) score += 5;
    return score;
}

static void consider(wchar_t *best, int *best_score, const wchar_t *path, const wchar_t *name, int depth) {
    int score = python_score(path, name, depth);
    if (score > *best_score) {
        *best_score = score;
        wcscpy(best, path);
    }
}

static void walk(const wchar_t *dir, int depth, wchar_t *best, int *best_score) {
    wchar_t pattern[PATH_CAP];
    WIN32_FIND_DATAW fd;
    HANDLE handle;
    if (depth > DEPTH_MAX) return;
    swprintf(pattern, PATH_CAP, L"%s\\*", dir);
    handle = FindFirstFileW(pattern, &fd);
    if (handle == INVALID_HANDLE_VALUE) return;
    do {
        wchar_t child[PATH_CAP];
        if (wcscmp(fd.cFileName, L".") == 0 || wcscmp(fd.cFileName, L"..") == 0) continue;
        swprintf(child, PATH_CAP, L"%s\\%s", dir, fd.cFileName);
        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
            walk(child, depth + 1, best, best_score);
        } else if (_wcsicmp(fd.cFileName, L"python.exe") == 0 || _wcsicmp(fd.cFileName, L"python3.exe") == 0) {
            consider(best, best_score, child, fd.cFileName, depth);
        }
    } while (FindNextFileW(handle, &fd));
    FindClose(handle);
}

static char *capture(const wchar_t *cmdline, const wchar_t *cwd, DWORD *code) {
    SECURITY_ATTRIBUTES sa;
    HANDLE read_handle = NULL;
    HANDLE write_handle = NULL;
    char *buf;
    DWORD got = 0;
    DWORD total = 0;
    sa.nLength = sizeof sa;
    sa.lpSecurityDescriptor = NULL;
    sa.bInheritHandle = TRUE;
    if (!CreatePipe(&read_handle, &write_handle, &sa, 0)) return NULL;
    SetHandleInformation(read_handle, HANDLE_FLAG_INHERIT, 0);
    if (!run_process(cmdline, cwd, 0, write_handle, (int *)code)) {
        CloseHandle(read_handle);
        CloseHandle(write_handle);
        return NULL;
    }
    CloseHandle(write_handle);
    buf = (char *)malloc(8192);
    if (!buf) fail(L"Out of memory.");
    while (total + 1 < 8192 && ReadFile(read_handle, buf + total, 8191 - total, &got, NULL) && got) {
        total += got;
    }
    buf[total] = 0;
    CloseHandle(read_handle);
    return buf;
}

static void split_lines(char *text, char **first, char **second) {
    char *nl = strchr(text, '\n');
    *first = text;
    *second = "";
    if (!nl) return;
    *nl = 0;
    *second = nl + 1;
    nl = strchr(*second, '\n');
    if (nl) *nl = 0;
    nl = strchr(*second, '\r');
    if (nl) *nl = 0;
    nl = strchr(*first, '\r');
    if (nl) *nl = 0;
}

static void write_stamp(const wchar_t *root, const char *digest) {
    wchar_t path[PATH_CAP];
    HANDLE file;
    DWORD written = 0;
    swprintf(path, PATH_CAP, L"%s\\runtime\\.requirements.sha256", root);
    file = CreateFileW(path, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (file == INVALID_HANDLE_VALUE) return;
    WriteFile(file, digest, (DWORD)strlen(digest), &written, NULL);
    WriteFile(file, "\n", 1, &written, NULL);
    CloseHandle(file);
}

int WINAPI WinMain(HINSTANCE inst, HINSTANCE prev, LPSTR cmd, int show) {
    wchar_t exe[PATH_CAP];
    wchar_t root[PATH_CAP];
    wchar_t runtime[PATH_CAP];
    wchar_t python[PATH_CAP];
    wchar_t play[PATH_CAP];
    wchar_t req[PATH_CAP];
    wchar_t check_cmd[PATH_CAP];
    wchar_t pip_cmd[PATH_CAP];
    wchar_t launch_cmd[PATH_CAP];
    wchar_t *quoted_py;
    wchar_t *quoted_root;
    wchar_t *quoted_play;
    wchar_t *quoted_req;
    DWORD code = 1;
    char *output;
    char *status;
    char *digest;
    int best_score = -1;
    int argc = 0;
    int i;
    int exit_code = 1;
    LPWSTR *argv;
    (void)inst;
    (void)prev;
    (void)cmd;
    (void)show;

    if (!GetModuleFileNameW(NULL, exe, PATH_CAP)) fail(L"Could not find the launcher path.");
    wcscpy(root, exe);
    {
        wchar_t *slash = wcsrchr(root, L'\\');
        if (!slash) fail(L"Could not find the launcher path.");
        *slash = 0;
    }
    swprintf(runtime, PATH_CAP, L"%s\\runtime", root);
    swprintf(play, PATH_CAP, L"%s\\play.py", root);
    swprintf(req, PATH_CAP, L"%s\\requirements.txt", root);
    python[0] = 0;
    walk(runtime, 0, python, &best_score);
    if (best_score < 0) {
        fail(L"WinPython was not found under the runtime folder. Put a WinPython build in runtime, then start the game again.");
    }
    if (GetFileAttributesW(req) == INVALID_FILE_ATTRIBUTES) {
        fail(L"requirements.txt is missing, so the game's dependencies cannot be installed.");
    }
    if (GetFileAttributesW(play) == INVALID_FILE_ATTRIBUTES) {
        fail(L"play.py is missing next to the launcher.");
    }

    quoted_py = quote_arg(python);
    quoted_root = quote_arg(root);
    quoted_req = quote_arg(req);
    check_cmd[0] = 0;
    append(check_cmd, PATH_CAP, quoted_py);
    append(check_cmd, PATH_CAP,
           L" -c \"import hashlib,pathlib,sys; root=pathlib.Path(sys.argv[1]); "
           L"digest=hashlib.sha256((root/'requirements.txt').read_bytes()).hexdigest(); "
           L"stamp=root/'runtime'/'.requirements.sha256'; missing=False\n"
           L"try:\n import numpy,pyglet,pyogg,moderngl,flask\n"
           L"except Exception:\n missing=True\n"
           L"text=stamp.read_text(encoding='utf-8').strip() if stamp.is_file() else ''\n"
           L"sys.stdout.write(('install' if missing or text!=digest else 'ready')+'\\n'+digest+'\\n')\" ");
    append(check_cmd, PATH_CAP, quoted_root);
    output = capture(check_cmd, root, &code);
    if (!output || code != 0) {
        fail(L"WinPython could not start. The copy under runtime may be incomplete.");
    }
    split_lines(output, &status, &digest);
    if (strcmp(status, "ready") != 0) {
        int pip_code = 1;
        pip_cmd[0] = 0;
        append(pip_cmd, PATH_CAP, quoted_py);
        append(pip_cmd, PATH_CAP, L" -m pip install --disable-pip-version-check -r ");
        append(pip_cmd, PATH_CAP, quoted_req);
        if (!run_process(pip_cmd, root, CREATE_NEW_CONSOLE, NULL, &pip_code) || pip_code != 0) {
            wchar_t ensure[PATH_CAP];
            ensure[0] = 0;
            append(ensure, PATH_CAP, quoted_py);
            append(ensure, PATH_CAP, L" -m ensurepip --upgrade");
            run_process(ensure, root, CREATE_NEW_CONSOLE, NULL, NULL);
            if (!run_process(pip_cmd, root, CREATE_NEW_CONSOLE, NULL, &pip_code) || pip_code != 0) {
                fail(L"Could not install the game's dependencies into WinPython.");
            }
        }
        write_stamp(root, digest);
    }

    {
        wchar_t pythonw[PATH_CAP];
        wchar_t *slash;
        wcscpy(pythonw, python);
        slash = wcsrchr(pythonw, L'\\');
        if (slash) {
            slash[1] = 0;
            wcscat(pythonw, L"pythonw.exe");
            if (GetFileAttributesW(pythonw) != INVALID_FILE_ATTRIBUTES) {
                free(quoted_py);
                quoted_py = quote_arg(pythonw);
            }
        }
    }

    quoted_play = quote_arg(play);
    launch_cmd[0] = 0;
    append(launch_cmd, PATH_CAP, quoted_py);
    append(launch_cmd, PATH_CAP, L" ");
    append(launch_cmd, PATH_CAP, quoted_play);
    argv = CommandLineToArgvW(GetCommandLineW(), &argc);
    if (argv) {
        for (i = 1; i < argc; ++i) {
            wchar_t *q = quote_arg(argv[i]);
            append(launch_cmd, PATH_CAP, L" ");
            append(launch_cmd, PATH_CAP, q);
            free(q);
        }
        LocalFree(argv);
    }
    if (!run_process(launch_cmd, root, 0, NULL, &exit_code)) {
        fail(L"WinPython could not start the game.");
    }
    free(quoted_py);
    free(quoted_root);
    free(quoted_req);
    free(quoted_play);
    free(output);
    ExitProcess(exit_code == 0 ? 0 : exit_code);
    return exit_code;
}
