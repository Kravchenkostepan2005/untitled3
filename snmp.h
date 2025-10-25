/*
 * Author: Stepan Kravchenko (xkravc03)
 */

#ifndef SNMP_H
#define SNMP_H

#include <stddef.h>
#include <stdint.h>:

typedef struct {
    int sockfd;
    int port;
    char target[256];
    char community[128];
    int timeout_ms;
    int retries;
    int verbose;
} snmp_client_t;

typedef enum {
    SNMP_TYPE_INTEGER = 0x02,
    SNMP_TYPE_OCTET_STRING = 0x04,
    SNMP_TYPE_NULL = 0x05,
    SNMP_TYPE_OBJECT_IDENTIFIER = 0x06,
    SNMP_TYPE_SEQUENCE = 0x30,
    SNMP_TYPE_TIMETICKS = 0x43,
} snmp_asn1_type_t;

typedef struct {
    uint32_t oid[64];
    size_t oid_len;
    uint8_t type;
    int64_t int_value;
    const uint8_t *str_value;
    size_t str_len;
} snmp_varbind_t;

int snmp_client_init(snmp_client_t *c, const char *target, int port, const char *community, int timeout_ms, int retries, int verbose);
void snmp_client_close(snmp_client_t *c);

// Perform SNMPv2c Get for one scalar OID (must end with .0). Returns 0 on success, fills vb. Returns -1 on error or timeout after retries.
int snmp_get(snmp_client_t *c, const uint32_t *oid, size_t oid_len, snmp_varbind_t *vb_out);

#endif
