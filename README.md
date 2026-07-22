# challenge-alura-agente
Agente corporativo con RAG para consultar documentos internos, desarrollado para el Challenge Alura Agente.

## Configuración local

1. Crea el entorno virtual:

   ```powershell
   python -m venv .venv
   ```

2. Actívalo en PowerShell:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

3. Instala las dependencias de desarrollo:

   ```powershell
   python -m pip install -r requirements-dev.txt
   ```

4. Copia `.env.example` como `.env` y configura `GOOGLE_API_KEY` localmente.

   ```powershell
   Copy-Item .env.example .env
   ```

El archivo `.env` no debe subirse al repositorio.
