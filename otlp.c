/*
 * Author: Stepan Kravchenko (xkravc03)
 */

#define _POSIX_C_SOURCE 200809L
#include "otlp.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <time.h>

int parse_http_url(const char *url, http_url_t *out)
{
	memset(out, 0, sizeof(*out));
	const char *p = strstr(url, "://");
	if (!p) return -1;
	size_t slen = (size_t)(p - url);
	if (slen >= sizeof(out->scheme)) return -1;
	memcpy(out->scheme, url, slen); out->scheme[slen] = '\0';
	p += 3;

	const char *slash = strchr(p, '/');
	const char *hostport_end = slash ? slash : p + strlen(p);
	const char *colon = NULL;
	for (const char *q = p; q < hostport_end; ++q) {
		if (*q == ':') { colon = q; break; }
	}

	if (colon) {
		size_t hlen = (size_t)(colon - p);
		if (hlen == 0 || hlen >= sizeof(out->host)) return -1;
		memcpy(out->host, p, hlen); out->host[hlen] = '\0';
		int port = atoi(colon + 1);
		if (port <= 0 || port > 65535) return -1;
		out->port = port;
	} else {
		size_t hlen = (size_t)(hostport_end - p);
		if (hlen == 0 || hlen >= sizeof(out->host)) return -1;
		memcpy(out->host, p, hlen); out->host[hlen] = '\0';
		out->port = (strcmp(out->scheme, "https") == 0) ? 443 : 80;
	}

	if (slash) {
		size_t plen = strlen(slash);
		if (plen >= sizeof(out->path)) return -1;
		memcpy(out->path, slash, plen + 1);
	} else {
		strcpy(out->path, "/");
	}
	return 0;
}

static int http_post_json(const http_url_t *u, const char *json, size_t json_len, int verbose)
{
	struct addrinfo hints; memset(&hints, 0, sizeof(hints));
	hints.ai_family = AF_UNSPEC;
	hints.ai_socktype = SOCK_STREAM;

	char portstr[16]; snprintf(portstr, sizeof(portstr), "%d", u->port);
	struct addrinfo *res = NULL;
	int rc = getaddrinfo(u->host, portstr, &hints, &res);
	if (rc != 0) { if (verbose) fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(rc)); return -1; }

	int sock = -1;
	for (struct addrinfo *ai = res; ai; ai = ai->ai_next) {
		sock = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
		if (sock < 0) continue;
		if (connect(sock, ai->ai_addr, ai->ai_addrlen) == 0) break;
		close(sock); sock = -1;
	}
	freeaddrinfo(res);
	if (sock < 0) return -1;

	char header[1024];
	int hn = snprintf(header, sizeof(header),
		"POST %s HTTP/1.1\r\nHost: %s\r\nContent-Type: application/json\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n",
		u->path, u->host, json_len);
	if (hn <= 0 || (size_t)hn >= sizeof(header)) { close(sock); return -1; }

	ssize_t w = send(sock, header, (size_t)hn, 0);
	if (w != (ssize_t)hn) { if (verbose) perror("send header"); close(sock); return -1; }
	if (json_len) {
		ssize_t wb = send(sock, json, json_len, 0);
		if (wb != (ssize_t)json_len) { if (verbose) perror("send body"); close(sock); return -1; }
	}

	size_t cap = 4096, len = 0;
	char *resp = (char*)malloc(cap);
	if (!resp) { close(sock); return -1; }

	for (;;) {
		if (len == cap) {
			cap *= 2;
			char *tmp = (char*)realloc(resp, cap);
			if (!tmp) { free(resp); close(sock); return -1; }
			resp = tmp;
		}
		ssize_t n = recv(sock, resp + len, cap - len, 0);
		if (n < 0) { if (verbose) perror("recv"); free(resp); close(sock); return -1; }
		if (n == 0) break;
		len += (size_t)n;
	}
	close(sock);

	if (verbose) {
		fwrite(resp, 1, len, stderr);
		fputc('\n', stderr);
	}

	// Parse status code
	int status = -1;
	if (len >= 12 && !memcmp(resp, "HTTP/", 5)) {
		// Find first space and parse the int that follows
		char *sp = memchr(resp, ' ', len);
		if (sp && (sp + 3) < resp + len) status = atoi(sp + 1);
	}
	free(resp);
	return status; // 2xx => success in caller
}

int otlp_export_gauge(const http_url_t *endpoint, const char *metric_name, const char *unit, double value, int verbose)
{
	char json[1024];
	struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
	unsigned long long tn = (unsigned long long)ts.tv_sec * 1000000000ULL + (unsigned long long)ts.tv_nsec;

	int n = snprintf(json, sizeof(json),
		"{\"resourceMetrics\":[{\"scopeMetrics\":[{\"metrics\":[{\"name\":\"%s\",\"unit\":\"%s\",\"gauge\":{\"dataPoints\":[{\"asDouble\":%.10g,\"timeUnixNano\":\"%llu\"}]}}]}]}]}",
		metric_name, unit ? unit : "", value, tn);
	if (n <= 0 || (size_t)n >= sizeof(json)) return -1;

	int status = http_post_json(endpoint, json, (size_t)n, verbose);
	if (verbose) fprintf(stderr, "OTLP status: %d\n\n", status);
	return (status >= 200 && status < 300) ? 0 : -1;
}
