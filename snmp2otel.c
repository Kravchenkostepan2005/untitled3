/*
 * Author: Stepan Kravchenko (xkravc03)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>
#include <getopt.h>

#include "oid.h"
#include "snmp.h"
#include "otlp.h"
#include "mapping.h"

typedef struct {
	char target[256];
	char community[128];
	char oids_file[256];
	char endpoint_url[512];
	int interval;
	int retries;
	int timeout_ms;
	int port;
	int verbose;
	char mapping_file[256];
} config_t;

static void usage(void)
{
	fprintf(stderr, "Usage: snmp2otel -t target [-C community] -o oids_file -e endpoint [-i interval] [-r retries] [-T timeout] [-p port] [-m mapping.json] [-v]\n");
}

static int load_oids(const char *path, uint32_t (*oids)[64], size_t *oid_lens, size_t *count)
{
	FILE *f = fopen(path, "rb");
	if (!f) { perror("open oids file"); return -1; }
	char line[512]; size_t n = 0;
	while (fgets(line, sizeof(line), f)) {
		char *p = line;
		while (*p == ' ' || *p == '\t') ++p;
		if (*p == '\0' || *p == '\n' || *p == '#' ) continue;
		char *nl = strchr(p, '\n'); if (nl) *nl = '\0';
		uint32_t tmp[64]; size_t len = oid_parse(p, tmp, 64);
		if (len == 0) { fprintf(stderr, "Invalid OID: %s\n", p); continue; }
		if (len == 0 || tmp[len-1] != 0) { fprintf(stderr, "OID must be scalar ending .0: %s\n", p); continue; }
		if (n < *count) {
			memcpy(oids[n], tmp, len * sizeof(uint32_t));
			oid_lens[n] = len;
			n++;
		}
	}
	fclose(f);
	*count = n;
	return 0;
}

int main(int argc, char **argv)
{
	config_t cfg; memset(&cfg, 0, sizeof(cfg));
	strcpy(cfg.community, "public");
	cfg.interval = 10; cfg.retries = 2; cfg.timeout_ms = 1000; cfg.port = 161; cfg.verbose = 0;

	int opt;
	while ((opt = getopt(argc, argv, "t:C:o:e:i:r:T:p:m:v")) != -1) {
		switch (opt) {
			case 't': strncpy(cfg.target, optarg, sizeof(cfg.target)-1); break;
			case 'C': strncpy(cfg.community, optarg, sizeof(cfg.community)-1); break;
			case 'o': strncpy(cfg.oids_file, optarg, sizeof(cfg.oids_file)-1); break;
			case 'e': strncpy(cfg.endpoint_url, optarg, sizeof(cfg.endpoint_url)-1); break;
			case 'i': cfg.interval = atoi(optarg); break;
			case 'r': cfg.retries = atoi(optarg); break;
			case 'T': cfg.timeout_ms = atoi(optarg); break;
			case 'p': cfg.port = atoi(optarg); break;
			case 'm': strncpy(cfg.mapping_file, optarg, sizeof(cfg.mapping_file)-1); break;
			case 'v': cfg.verbose = 1; break;
			default: usage(); return 1;
		}
	}
	if (cfg.target[0] == '\0' || cfg.oids_file[0] == '\0' || cfg.endpoint_url[0] == '\0' || cfg.interval <= 0) { usage(); return 1; }

	http_url_t endpoint; if (parse_http_url(cfg.endpoint_url, &endpoint) != 0) { fprintf(stderr, "Invalid endpoint URL\n"); return 1; }

	mapping_table_t mapping = {0};
	if (cfg.mapping_file[0]) {
		if (mapping_load(cfg.mapping_file, &mapping) != 0) fprintf(stderr, "Failed to load mapping file, proceeding without mapping.\n");
	}

	uint32_t oids[128][64]; size_t oid_lens[128]; size_t count = 128;
	if (load_oids(cfg.oids_file, oids, oid_lens, &count) != 0 || count == 0) { fprintf(stderr, "No valid OIDs loaded\n"); mapping_free(&mapping); return 1; }

	snmp_client_t snmp;
	if (snmp_client_init(&snmp, cfg.target, cfg.port, cfg.community, cfg.timeout_ms, cfg.retries, cfg.verbose) != 0) { mapping_free(&mapping); return 1; }

	for (;;) {
		for (size_t i = 0; i < count; ++i) {
			snmp_varbind_t vb;
			if (snmp_get(&snmp, oids[i], oid_lens[i], &vb) != 0) {
				fprintf(stderr, "SNMP get failed for index %zu\n", i);
				continue;
			}
			char oid_str[256]; oid_to_string(vb.oid, vb.oid_len, oid_str, sizeof(oid_str));
			const mapping_entry_t *me = mapping_find(&mapping, oid_str);
			char name[128]; char unit[32] = "";
			if (me && me->name[0]) { strncpy(name, me->name, sizeof(name)-1); name[sizeof(name)-1] = '\0'; strncpy(unit, me->unit, sizeof(unit)-1); }
			else { snprintf(name, sizeof(name), "snmp.%s", oid_str); }

			double value = 0.0;

			if (vb.type == SNMP_TYPE_TIMETICKS) {
				value = (double)vb.int_value * 10.0;
				if (!(me && me->unit[0])) {
					strncpy(unit, "ms", sizeof(unit) - 1);
					unit[sizeof(unit) - 1] = '\0';
				}
			} else if (vb.type == SNMP_TYPE_INTEGER) {
				value = (double)vb.int_value;
			} else if (vb.type == SNMP_TYPE_OCTET_STRING) {
				char tmp[128];
				size_t n = vb.str_len < sizeof(tmp) - 1 ? vb.str_len : sizeof(tmp) - 1;
				memcpy(tmp, vb.str_value, n);
				tmp[n] = '\0';
				value = atof(tmp);
			} else {
				if (cfg.verbose) fprintf(stderr, "Unsupported type 0x%02x for %s\n", vb.type, oid_str);
				continue;
			}
			if (otlp_export_gauge(&endpoint, name, unit, value, cfg.verbose) != 0) {
				fprintf(stderr, "OTLP export failed for %s\n", name);
			}
		}
		if (cfg.verbose) fprintf(stderr, "sleep %d sec\n", cfg.interval);
		sleep((unsigned int)cfg.interval);
	}

	snmp_client_close(&snmp);
	mapping_free(&mapping);
	return 0;
}