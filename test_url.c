/*
 * Author: Stepan Kravchenko (xkravc03)
 */

#include <stdio.h>
#include <string.h>
#include "otlp.h"

int test_url(void)
{
    http_url_t u;
    if (parse_http_url("http://localhost:4318/v1/metrics", &u) != 0) { printf("parse url failed\n"); return 1; }
    if (strcmp(u.scheme, "http") != 0 || strcmp(u.host, "localhost") != 0 || u.port != 4318 || strcmp(u.path, "/v1/metrics") != 0) { printf("url fields mismatch\n"); return 1; }
    return 0;
}
