# Regimens Assembler
Pipeline for assembling regimens from HemOnc datasets (sigs table)

For details checkout `assets/Assembler_main.md`

# Setup

0. You would need to setup postgres connection first to access Athena mirror resources.

1. Use requirements.txt to setup env. Checkout `.env.template`.

# Run pipeline

```
./MAIN.sh -out output-assembled
```
