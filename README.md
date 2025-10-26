# ISA 2025/2026 – snmp2otel

- **Jméno a login**: Stepan Kravchenko (xkravc03)
- **Datum**: 26.10.2025.

## Popis
Program snmp2otel periodicky dotazuje SNMP agenta (SNMPv2c, pouze Get pro skalární OID .0) a naměřené hodnoty exportuje jako OpenTelemetry Metrics typu Gauge přes OTLP/HTTP (JSON) na zadaný endpoint. Aplikace se překládá pomocí Makefile, nevyžaduje root ani instalaci knihoven do systému, je modulární a obsahuje základní testy.

## Příklad spuštění
```bash
./snmp2otel -t isa.fit.vutbr.cz -C public -o ./oids.txt \
            -e http://127.0.0.1:4319/v1/metrics \
            -i 5 -T 3000 -r 3 -v
