# hungarian-algorithm-SYCL
# Guida per l'esecuzione su Tiber AI Cloud e Sistema Locale

## SU TIBER AI CLOUD (Jupyter)

1. Effettuare il login nella piattaforma Jupyter.
2. Aggiungere i file:
   - `q`
   - `run.sh`
   - `SYCL_Hungarian_Algorithm.cpp`
3. Lanciare `run.sh` usando il seguente comando:

   ```bash
   ! chmod 755 q; chmod 755 run.sh; if [ -x "$(command -v qsub)" ]; then ./q run.sh; else ./run.sh; fi
   ```

## SU SISTEMA LOCALE

1. Assicurarsi di avere i seguenti file:
   - `q`
   - `run.sh`
   - `SYCL_Hungarian_Algorithm.cpp`

2. Per compilare il codice, lanciare su terminale uno dei seguenti comandi:

   - **GPU (NVIDIA):**
     ```bash
     icpx -o3 -fsycl -fsycl-targets=nvptx64-nvidia-cuda ./src/SYCL_Hungarian_Algorithm.cpp -o ./src/SYCL_Hungarian_Algorithm
     ```
   
   - **CPU:**
     ```bash
     icpx -o3 -fsycl -fsycl-targets=spir64_x86_64 ./src/SYCL_Hungarian_Algorithm.cpp -o ./src/SYCL_Hungarian_Algorithm
     ```

