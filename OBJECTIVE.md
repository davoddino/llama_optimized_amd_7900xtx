# Objective

L'obiettivo e implementare un vero megakernel di decode per Qwen3.6 su AMD Radeon RX 7900 XTX: tutti i 40 layer devono essere fusi in un solo dispatch GPU.

Il target non e fondere un singolo layer. Il target e attraversare l'intero blocco dei 40 layer dentro un unico kernel persistente, senza round-trip CPU, senza sequenze di launch per layer e senza loop generici nel path critico. Il dataflow deve essere fisico, esplicito e specializzato per la forma esatta del modello e per l'architettura RDNA3/gfx1100.

Il riferimento concettuale e Lucebox: un megakernel hand-tuned che mette tutti i layer del modello target in un singolo dispatch, usando esecuzione persistente e sincronizzazione cooperativa per eliminare i launch intermedi.

## Target Prestazionale

Il decode deve superare i 900 token/s sulla RX 7900 XTX.

La misura rilevante e il decode a token singolo, con modello gia caricato su GPU, KV cache residente e sampling compatibile con esecuzione device-resident.

## Direzione Tecnica

Il megakernel deve sfruttare al massimo la GPU 7900 XTX, assumendo conoscenza dettagliata di:

- RDNA3/gfx1100
- wavefront a 32 lane
- occupancy effettiva sui Compute Unit
- scheduling dei CU e uso di persistent blocks
- gerarchia memoria, LDS, cache e accessi globali
- costo reale di launch, wait, barrier, host sync e device-to-host readback
- layout dei tensori quantizzati, KV cache e buffer runtime di llama.cpp

Il lavoro deve restare il piu vicino possibile alle unita computazionali. I dati devono essere caricati, riusati e consumati localmente, mantenendoli in registri, LDS o cache quando il dataflow lo consente. La memoria globale deve essere usata solo dove serve al confine reale tra stadi o per persistenza necessaria.

## Vincoli Del Path Critico

Il path di decode deve eliminare ogni bottleneck evitabile:

- un solo dispatch GPU per attraversare i 40 layer
- nessun host sync nel percorso caldo
- nessun readback dei logits grezzi quando il backend sampling e attivo
- nessun launch per layer
- nessun loop generico che interpreti il grafo durante il decode
- nessuna materializzazione intermedia non necessaria in memoria globale
- nessuna wait o barrier oltre a quelle richieste dal dataflow cooperativo
- nessun passaggio CPU per routing, sampling o controllo quando puo restare su GPU

Gli scambi di dati con la memoria devono tendere a zero rispetto al lavoro computazionale utile. I dati caricati devono essere consumati subito, riusati localmente e scritti solo quando il risultato e necessario per il passaggio successivo del megakernel o per l'output finale.

## Ambito Del Megakernel

Il megakernel deve coprire l'intero decode Qwen3.6 attraverso tutti i 40 layer:

- normalizzazioni
- proiezioni QKV
- attention layer
- recurrent/GDN layer
- routing MoE
- proiezioni gate/up/down
- accumuli, residui e stati intermedi
- aggiornamento della KV cache
- output finale pronto per sampling device-resident

Il dataflow deve essere scritto come programma specializzato per il modello, non come esecuzione del grafo generale. Ogni layer puo avere codice specializzato, ma il passaggio attraverso i 40 layer deve restare dentro un unico megakernel.

## Stato Atteso

Quando il path e attivo, i log devono indicare chiaramente che il megakernel sta eseguendo dataflow numerico reale per i 40 layer, non solo un contratto, uno scaffold, una validazione del grafo o una fusione parziale.

Il criterio di successo e semplice: output corretto, chat funzionante, decode sopra 900 token/s, un solo dispatch per token e assenza di bottleneck di sync, wait o traffico memoria evitabile nel path caldo.
