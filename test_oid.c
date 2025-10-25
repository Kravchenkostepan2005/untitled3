/*
 * Author: Stepan Kravchenko (xkravc03)
 */

#include <stdio.h>
#include <string.h>
#include "oid.h"

int test_oid(void)
{
    uint32_t out[16];
    size_t n = oid_parse("1.3.6.1.2.1.1.3.0", out, 16);
    if (n == 0) { printf("oid_parse failed\n"); return 1; }
    char buf[128];
    size_t m = oid_to_string(out, n, buf, sizeof(buf));
    if (m == 0) { printf("oid_to_string failed\n"); return 1; }
    if (strcmp(buf, "1.3.6.1.2.1.1.3.0") != 0) { printf("oid roundtrip mismatch: %s\n", buf); return 1; }
    return 0;
}
