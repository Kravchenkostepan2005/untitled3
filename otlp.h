#ifndef OTLP_H
#define OTLP_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    char scheme[8];
    char host[256];
    int port;
    char path[512];
} http_url_t;

typedef struct {
    char name[128];
    char unit[32];
} metric_info_t;

// Minimal URL parser for http(s)://host[:port]/path
int parse_http_url(const char *url, http_url_t *out);

// Export a single gauge metric in OTLP/HTTP JSON. Returns 0 on success.
int otlp_export_gauge(const http_url_t *endpoint, const char *metric_name, const char *unit, double value, int verbose);

#endif
