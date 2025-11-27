# Regimens Assembler
Pipeline for assembling regimens from HemOnc datasets (sigs table)

For strategy and details see `assets/Assembler_main.md`

## Setup

- You need to setup connection with OMOP CDM DB first. 
See `.env.template`.

- Install environment from `requirements.txt`

## Run 

Create regimens with the following command:

```
./MAIN.sh -out output-assembled
```
