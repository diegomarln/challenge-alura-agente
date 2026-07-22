# Despliegue de Streamlit en OCI Compute

Estos archivos preparan la aplicación para ejecutarse con Docker Compose. La
imagen contiene únicamente la aplicación y sus dependencias de producción; la
configuración se recibe desde un archivo `.env` local no versionado.

## Requisitos

- Docker Engine.
- Plugin Docker Compose.
- Repositorio clonado en la instancia.
- Archivo `.env` local, no versionado, con la configuración requerida.

`GOOGLE_API_KEY` se configura únicamente en el servidor. No muestres el
contenido de `.env`, no lo agregues a Git y no copies tu archivo local desde
otro equipo mediante el repositorio.

## Operación

Construir e iniciar:

```bash
docker compose up -d --build
```

Consultar el estado:

```bash
docker compose ps
```

Ver los últimos registros:

```bash
docker compose logs --tail=100 app
```

Seguir los registros:

```bash
docker compose logs -f app
```

Comprobar el healthcheck local:

```bash
curl --fail http://localhost:8501/_stcore/health
```

Reiniciar la aplicación:

```bash
docker compose restart app
```

Detener los contenedores sin borrar el índice persistido:

```bash
docker compose down
```

Detener y borrar también el índice persistido:

```bash
docker compose down -v
```

Advertencia: `docker compose down -v` elimina el volumen del índice vectorial.
Permite el puerto 8501 tanto en OCI como en el firewall del sistema antes de
acceder a la aplicación.

## Verificación funcional

Realiza estas consultas tras confirmar que el servicio está saludable:

- ¿Cuántos días de teletrabajo se permiten por semana? Debe indicar 2 días
  remotos y 3 presenciales.
- ¿Cuál es el monto máximo mensual que se puede reembolsar por internet? Debe
  indicar 35 EUR.
- ¿Cuál será el precio del dólar la próxima semana? Debe devolver el fallback
  sin fuentes.
