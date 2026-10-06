/* Shim over mingw's wrl.h: the samples' DXSampleHelper.h needs
 * Microsoft::WRL::Wrappers::FileHandle, which mingw-w64 does not provide. */
#pragma once
#include_next <wrl.h>
#include <windows.h>
namespace Microsoft { namespace WRL { namespace Wrappers {
class FileHandle {
   HANDLE h_;
public:
   explicit FileHandle(HANDLE h = INVALID_HANDLE_VALUE) : h_(h) {}
   ~FileHandle() { if (IsValid()) CloseHandle(h_); }
   FileHandle(const FileHandle &) = delete;
   FileHandle &operator=(const FileHandle &) = delete;
   HANDLE Get() const { return h_; }
   bool IsValid() const { return h_ != INVALID_HANDLE_VALUE && h_ != nullptr; }
};
} } }
