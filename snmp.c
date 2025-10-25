/*
 * Author: Stepan Kravchenko (xkravc03)
 */

#include "snmp.h"
#include "ber.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <time.h>

static void dump_hex(const char *prefix, const uint8_t *data, size_t len)
{
    if (!data || len == 0) return;
    fprintf(stderr, "%s", prefix ? prefix : "");
    for (size_t i = 0; i < len; ++i) {
        fprintf(stderr, "%02X%s", data[i], (i + 1 == len) ? "" : " ");
    }
    fprintf(stderr, "\n");
}

static uint32_t rand32(void)
{
    uint32_t value;
    FILE *f = fopen("/dev/urandom", "rb");
    if (f) {
        if (fread(&value, 1, sizeof(value), f) != sizeof(value)) {
            value = (uint32_t)time(NULL);
        }
        fclose(f);
    } else {
        value = (uint32_t)time(NULL);
    }
    return value;
}

int snmp_client_init(snmp_client_t *client,
                     const char *target,
                     int port,
                     const char *community,
                     int timeout_ms,
                     int retries,
                     int verbose)
{
    memset(client, 0, sizeof(*client));
    strncpy(client->target, target ? target : "", sizeof(client->target) - 1);
    strncpy(client->community, community ? community : "public", sizeof(client->community) - 1);
    client->port = port;
    client->timeout_ms = timeout_ms;
    client->retries = retries;
    client->verbose = verbose;
    client->sockfd = -1;
    return 0;
}

void snmp_client_close(snmp_client_t *client)
{
    if (client->sockfd >= 0) {
        close(client->sockfd);
    }
    client->sockfd = -1;
}

static int send_and_recv(snmp_client_t *client,
                         const uint8_t *packet,
                         size_t packet_len,
                         uint8_t *response,
                         size_t response_cap)
{
    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family   = AF_UNSPEC;     // IPv6 a IPv4
    hints.ai_socktype = SOCK_DGRAM;
    hints.ai_protocol = IPPROTO_UDP;
    hints.ai_flags    = AI_ADDRCONFIG;

    char port_str[16];
    snprintf(port_str, sizeof(port_str), "%d", client->port);

    struct addrinfo *res = NULL;
    int rc = getaddrinfo(client->target, port_str, &hints, &res);
    if (rc != 0) {
        fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(rc));
        return -1;
    }

    int result = -1;
    for (struct addrinfo *ai = res; ai; ai = ai->ai_next) {
        int sock = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (sock < 0) {
            continue;
        }

        struct timeval tv;
        tv.tv_sec  = client->timeout_ms / 1000;
        tv.tv_usec = (client->timeout_ms % 1000) * 1000;
        setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

        ssize_t sent = sendto(sock, packet, packet_len, 0, ai->ai_addr, ai->ai_addrlen);
        if (sent < 0) {
            close(sock);
            continue;
        }

        ssize_t received = recvfrom(sock, response, response_cap, 0, NULL, NULL);
        if (received >= 0) {
            result = (int)received;
            close(sock);
            break;
        }

        close(sock);
    }

    freeaddrinfo(res);
    return result;
}

int snmp_get(snmp_client_t *client,
             const uint32_t *oid,
             size_t oid_len,
             snmp_varbind_t *vb_out)
{
    uint8_t message_buf[1500];
    ber_buf_t message;
    ber_buf_init(&message, message_buf, sizeof(message_buf));

    // SNMP Message (SEQUENCE)
    size_t message_len_pos;
    if (ber_start_sequence(&message, &message_len_pos) != 0) {
        return -1;
    }

    // version = 1 (v2c)
    if (ber_encode_integer(&message, 1) != 0) {
        return -1;
    }

    // community
    if (ber_encode_octet_string(&message, (const uint8_t *)client->community, strlen(client->community)) != 0) {
        return -1;
    }

    // VarBind = SEQUENCE { name OID, value NULL }
    uint8_t varbind_storage[512];
    ber_buf_t varbind;
    ber_buf_init(&varbind, varbind_storage, sizeof(varbind_storage));
    size_t varbind_len_pos;
    if (ber_start_sequence(&varbind, &varbind_len_pos) != 0) {
        return -1;
    }
    if (ber_encode_oid(&varbind, oid, oid_len) != 0) {
        return -1;
    }
    uint8_t null_tag = 0x05, null_len = 0x00;
    if (ber_buf_write(&varbind, &null_tag, 1) != 1) return -1;
    if (ber_buf_write(&varbind, &null_len, 1) != 1) return -1;
    if (ber_end_sequence(&varbind, varbind_len_pos) != 0) {
        return -1;
    }

    // VarBindList = SEQUENCE of VarBind
    uint8_t vbl_storage[1024];
    ber_buf_t vbl;
    ber_buf_init(&vbl, vbl_storage, sizeof(vbl_storage));
    size_t vbl_len_pos;
    if (ber_start_sequence(&vbl, &vbl_len_pos) != 0) {
        return -1;
    }
    if (ber_buf_write(&vbl, varbind.data, varbind.len) != varbind.len) {
        return -1;
    }
    if (ber_end_sequence(&vbl, vbl_len_pos) != 0) {
        return -1;
    }

    // GetRequest-PDU [0] IMPLICIT (0xA0)
    uint8_t pdu_content_storage[1024];
    ber_buf_t pdu_content;
    ber_buf_init(&pdu_content, pdu_content_storage, sizeof(pdu_content_storage));
    int32_t request_id = (int32_t)(rand32() & 0x7FFFFFFF);
    if (ber_encode_integer(&pdu_content, request_id) != 0) return -1;
    if (ber_encode_integer(&pdu_content, 0) != 0) return -1; // error-status
    if (ber_encode_integer(&pdu_content, 0) != 0) return -1; // error-index
    if (ber_buf_write(&pdu_content, vbl.data, vbl.len) != vbl.len) return -1;

    if (ber_encode_tlv(&message, 0xA0, pdu_content.data, pdu_content.len) != 0) {
        return -1;
    }
    if (ber_end_sequence(&message, message_len_pos) != 0) {
        return -1;
    }

    if (client->verbose) {
        fprintf(stderr, "SNMP GET send len=%zu\n", message.len);
        dump_hex("send hex: ", message.data, message.len);
    }

    uint8_t resp_buf[1500];
    for (int attempt = 0; attempt <= client->retries; ++attempt) {
        int rcv_len = send_and_recv(client, message.data, message.len, resp_buf, sizeof(resp_buf));
        if (rcv_len < 0) {
            if (client->verbose) {
                fprintf(stderr, "timeout or recv error, attempt %d/%d\n", attempt + 1, client->retries + 1);
            }
            continue;
        }

        // Message (SEQUENCE)
        size_t off = 0;
        uint8_t mtag; const uint8_t *mval; size_t mlen;
        if (ber_decode_tlv(resp_buf, (size_t)rcv_len, &off, &mtag, &mval, &mlen) != 0 || mtag != 0x30) {
            fprintf(stderr, "invalid SNMP message\n");
            return -1;
        }
        size_t moff = (size_t)(mval - resp_buf);

        // version
        int64_t version_value = 0;
        if (ber_decode_integer(resp_buf, (size_t)rcv_len, &moff, &version_value) != 0) {
            fprintf(stderr, "SNMP version missing\n");
            return -1;
        }
        if (client->verbose) fprintf(stderr, "decode: version=%lld\n", (long long)version_value);
        if (version_value != 1) {
            fprintf(stderr, "SNMP version mismatch\n");
            return -1;
        }

        // community
        const uint8_t *comm_ptr; size_t comm_len = 0;
        if (ber_decode_octet_string(resp_buf, (size_t)rcv_len, &moff, &comm_ptr, &comm_len) != 0) {
            fprintf(stderr, "SNMP community missing\n");
            return -1;
        }
        if (client->verbose) fprintf(stderr, "decode: community len=%zu\n", comm_len);

        // GetResponse-PDU (0xA2)
        uint8_t pdu_tag; const uint8_t *pdu_val; size_t pdu_len;
        if (ber_decode_tlv(resp_buf, (size_t)rcv_len, &moff, &pdu_tag, &pdu_val, &pdu_len) != 0) {
            fprintf(stderr, "PDU missing\n");
            return -1;
        }
        if (client->verbose) fprintf(stderr, "decode: PDU tag=0x%02X\n", pdu_tag);
        if (pdu_tag != 0xA2) {
            fprintf(stderr, "unexpected PDU tag\n");
            return -1;
        }

        // parse inside PDU
        size_t poff = (size_t)(pdu_val - resp_buf);

        int64_t rid = 0, err_status = 0, err_index = 0;
        if (ber_decode_integer(resp_buf, (size_t)rcv_len, &poff, &rid) != 0) return -1;
        if (ber_decode_integer(resp_buf, (size_t)rcv_len, &poff, &err_status) != 0) return -1;
        if (ber_decode_integer(resp_buf, (size_t)rcv_len, &poff, &err_index) != 0) return -1;
        if (client->verbose) fprintf(stderr, "decode: rid=%lld errst=%lld erridx=%lld\n",
                                     (long long)rid, (long long)err_status, (long long)err_index);
        if (rid != request_id) {
            if (client->verbose) fprintf(stderr, "request-id mismatch\n");
            continue; // retry
        }

        // VarBindList (SEQUENCE)
        uint8_t list_tag; const uint8_t *list_val; size_t list_len;
        if (ber_decode_tlv(resp_buf, (size_t)rcv_len, &poff, &list_tag, &list_val, &list_len) != 0 || list_tag != 0x30) {
            fprintf(stderr, "invalid VarBindList\n");
            return -1;
        }
        if (client->verbose) fprintf(stderr, "decode: vbl len=%zu\n", list_len);

        // first VarBind
        size_t loff = (size_t)(list_val - resp_buf);
        uint8_t vb_tag; const uint8_t *vb_val; size_t vb_len2;
        if (ber_decode_tlv(resp_buf, (size_t)rcv_len, &loff, &vb_tag, &vb_val, &vb_len2) != 0 || vb_tag != 0x30) {
            fprintf(stderr, "invalid VarBind\n");
            return -1;
        }

        // name OID
        size_t vboff = (size_t)(vb_val - resp_buf);
        uint32_t resp_oid[64]; size_t resp_oid_len = 0;
        if (ber_decode_oid(resp_buf, (size_t)rcv_len, &vboff, resp_oid, &resp_oid_len, 64) != 0) {
            fprintf(stderr, "invalid OID\n");
            return -1;
        }

        // value
        uint8_t value_tag; const uint8_t *value_ptr; size_t value_len;
        if (ber_decode_tlv(resp_buf, (size_t)rcv_len, &vboff, &value_tag, &value_ptr, &value_len) != 0) {
            fprintf(stderr, "missing value\n");
            return -1;
        }
        if (client->verbose) fprintf(stderr, "decode: value tag=0x%02X len=%zu\n", value_tag, value_len);

        // fill output
        vb_out->oid_len = resp_oid_len;
        for (size_t i = 0; i < resp_oid_len; ++i) vb_out->oid[i] = resp_oid[i];
        vb_out->type = value_tag;
        vb_out->str_value = value_ptr;
        vb_out->str_len = value_len;
        vb_out->int_value = 0;

        if (value_tag == 0x02 || value_tag == 0x43) { // INTEGER or TimeTicks
            int64_t decoded = 0;
            if (value_len > 0 && value_len <= 8) {
                int64_t accum = (value_ptr[0] & 0x80) ? -1 : 0;
                for (size_t k = 0; k < value_len; ++k) accum = (accum << 8) | value_ptr[k];
                decoded = accum;
            }
            vb_out->int_value = decoded;
        }

        return 0;
    }

    return -1;
}
