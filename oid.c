/*
 * Author: Stepan Kravchenko (xkravc03)
 */

#include "oid.h"
#include <string.h>
#include <ctype.h>

size_t oid_parse(const char *s, uint32_t *out, size_t max_len)
{
    size_t count = 0;
    uint64_t val = 0;
    int in_digit = 0;
    for (const char *p = s; ; ++p) {
        if (*p >= '0' && *p <= '9') {
            in_digit = 1;
            val = val * 10 + (uint64_t)(*p - '0');
            if (val > 0xFFFFFFFFULL) return 0;
        } else if (*p == '.' || *p == '\0') {
            if (!in_digit) return 0;
            if (count >= max_len) return 0;
            out[count++] = (uint32_t)val;
            val = 0; in_digit = 0;
            if (*p == '\0') break;
        } else if (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') {
            return 0;
        } else {
            return 0;
        }
    }
    return count;
}

size_t oid_to_string(const uint32_t *oid, size_t len, char *buf, size_t bufsize)
{
    if (bufsize == 0) return 0;
    size_t written = 0;
    for (size_t i = 0; i < len; ++i) {
        char tmp[16];
        int n = 0;
        unsigned int v = oid[i];
        do { tmp[n++] = (char)('0' + (v % 10)); v /= 10; } while (v);
        if (i != 0) {
            if (written + 1 >= bufsize) return 0;
            buf[written++] = '.';
        }
        if (written + (size_t)n >= bufsize) return 0;
        for (int j = n - 1; j >= 0; --j) buf[written++] = tmp[j];
    }
    if (written >= bufsize) return 0;
    buf[written] = '\0';
    return written;
}
