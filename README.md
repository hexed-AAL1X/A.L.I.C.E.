<a id="readme-top"></a>

<img src="https://github.com/AnderMendoza/AnderMendoza/raw/main/assets/line-neon.gif" width="100%">

<p align="center">
  <img alt="GitHub Repo contributors" src="https://img.shields.io/github/contributors/hexed-AAL1X/A.L.I.C.E-front?style=for-the-badge">&nbsp;
  <img alt="GitHub Repo forks" src="https://img.shields.io/github/forks/hexed-AAL1X/A.L.I.C.E-front?style=for-the-badge">&nbsp;
  <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/hexed-AAL1X/A.L.I.C.E-front?style=for-the-badge">&nbsp;
  <img alt="GitHub Repo issues" src="https://img.shields.io/github/issues/hexed-AAL1X/A.L.I.C.E-front?style=for-the-badge">&nbsp;
</p>

<br>

<div align="center">
  <img src="public/alice-eye.webp" alt="A.L.I.C.E." width="120" />
  <h3 align="center">A.L.I.C.E. Frontend</h3>
  <p align="center">
    Frontend web (Next.js) del asistente LLM A.L.I.C.E. — chat, tema claro/oscuro y splash de carga.
    <br>
    <a href="https://github.com/hexed-AAL1X/A.L.I.C.E-front"><strong>Explorar repositorio »</strong></a>
    <br><br>
    <a href="https://github.com/hexed-AAL1X/A.L.I.C.E-front">Ver código</a>
    ·
    <a href="https://github.com/hexed-AAL1X/A.L.I.C.E-front/issues/new?labels=bug">Reportar bug</a>
    ·
    <a href="https://github.com/hexed-AAL1X/A.L.I.C.E-front/issues/new?labels=enhancement">Pedir feature</a>
  </p>
</div>

<details>
  <summary>Tabla de contenidos</summary>
  <ol>
    <li><a href="#about-the-project">About the project</a></li>
    <li><a href="#built-with">Built with</a></li>
    <li><a href="#important-notices">Important notices</a></li>
    <li>
      <a href="#getting-started">Getting started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#available-scripts">Available scripts</a></li>
      </ul>
    </li>
    <li><a href="#environments">Environments</a></li>
    <li><a href="#deployment">Deployment</a></li>
    <li>
      <a href="#contributing">Contributing</a>
      <ul>
        <li><a href="#top-contributors">Top contributors</a></li>
      </ul>
    </li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>
<br>

<a id="about-the-project"></a>***About the project***
<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif">

<p align="center">
  <img src="public/alice-logo.webp" alt="A.L.I.C.E. wordmark" width="420" />
</p>

**A.L.I.C.E.** (frontend) es la interfaz de chat del asistente LLM.

Incluye:

- Chat con streaming hacia el backend Flask.
- Sidebar de conversaciones, tema claro/oscuro y wordmark metálico.
- Splash de carga sincronizado con el health/warmup del backend.
- Micrófono del navegador + TTS del cliente (sin STT/TTS en el server).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<a id="built-with"></a>***Built with***
<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif">

- ![Next.js](https://img.shields.io/badge/Next.js-14.0.4-000000?style=for-the-badge&logo=nextdotjs&logoColor=white)
- ![React](https://img.shields.io/badge/React-18.2-61DAFB?style=for-the-badge&logo=react&logoColor=black)
- ![TypeScript](https://img.shields.io/badge/TypeScript-5-3178C6?style=for-the-badge&logo=typescript&logoColor=white)
- ![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-3.3-06B6D4?style=for-the-badge&logo=tailwindcss&logoColor=white)
- ![Framer Motion](https://img.shields.io/badge/Framer_Motion-11-0055FF?style=for-the-badge&logo=framer&logoColor=white)
- ![Axios](https://img.shields.io/badge/Axios-1.6-5A29E4?style=for-the-badge&logo=axios&logoColor=white)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<a id="important-notices"></a>***Important notices***
<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif">

> [!NOTE]
> No necesitas instalar nada global aparte de Node.js.
>
> Usa `npm run dev` para levantar el servidor local.

> [!IMPORTANT]
> Este repo es el **frontend**. Para chatear necesitas el backend de A.L.I.C.E. corriendo y configurar `NEXT_PUBLIC_API_URL` en `.env.local`.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<a id="getting-started"></a>***Getting started***
<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif">

<a id="prerequisites"></a>

### Prerequisites

- Node.js (recomendado: LTS)
- npm
- Backend A.L.I.C.E. en `http://localhost:5000` (o la URL que configures)

<a id="installation"></a>

### Installation

1) Clonar el repositorio

```bash
git clone https://github.com/hexed-AAL1X/A.L.I.C.E-front.git
cd A.L.I.C.E-front
```

2) Instalar dependencias

```bash
npm install
```

3) Configurar entorno

```bash
cp .env.local.example .env.local
# o crea .env.local con:
# NEXT_PUBLIC_API_URL=http://localhost:5000
```

4) Ejecutar en modo desarrollo

```bash
npm run dev
```

5) Abrir en el navegador

- `http://localhost:3000/`

<a id="available-scripts"></a>

### Available scripts

```bash
npm run dev      # next dev
npm run build    # build producción
npm run start    # next start
npm run lint     # eslint
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<a id="environments"></a>***Environments***
<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif">

Variables en `.env.local` (Next.js):

| Variable | Descripción | Default |
| --- | --- | --- |
| `NEXT_PUBLIC_API_URL` | URL base del backend Flask | `http://localhost:5000` |

El front llama, entre otros, a `/api/health`, `/api/warmup` y `/api/chat/stream`.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<a id="deployment"></a>***Deployment***
<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif">

### Render (Web Service)

- **Root directory:** `frontend` (si el monorepo incluye más carpetas) o la raíz de este repo
- **Build command:**

```bash
npm ci && npm run build
```

- **Start command:**

```bash
npm run start -- -p $PORT
```

- **Env:**

```bash
NEXT_PUBLIC_API_URL=https://TU-BACKEND.onrender.com
```

> [!NOTE]
> En el free tier de Render el servicio duerme tras inactividad; el splash del front espera a `/api/health`.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<a id="contributing"></a>***Contributing***
<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif">

Contribuciones bienvenidas.

1) Fork del proyecto  
2) Crear una rama (`git checkout -b feature/nueva-feature`)  
3) Commit (`git commit -m "Add: ..."`)  
4) Push (`git push origin feature/nueva-feature`)  
5) Pull Request

<a id="top-contributors"></a>

### Top contributors

<div align="center">

<table>
  <tr>
    <td align="center" width="160">
      <a href="https://github.com/ShoterBroXD">
        <img src="https://avatars.githubusercontent.com/u/114694812?v=4" width="88" height="88" alt="Diego Melendez" style="border-radius:50%;" /><br />
        <b>Diego Melendez</b><br />
        <sub>@ShoterBroXD</sub>
      </a>
    </td>
    <td align="center" width="160">
      <a href="https://github.com/SebasTM502">
        <img src="https://avatars.githubusercontent.com/u/206435498?v=4" width="88" height="88" alt="SebasTM502" style="border-radius:50%;" /><br />
        <b>SebasTM502</b><br />
        <sub>@SebasTM502</sub>
      </a>
    </td>
  </tr>
</table>

</div>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<a id="contact"></a>***Contact***
<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif">

<p align="center">
  <a href="mailto:hexed_aal1x.ops@proton.me"><img src="https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white&color=black" /></a>
  <a href="https://www.instagram.com/hexed_aal1x"><img src="https://img.shields.io/badge/instagram-%2312100E.svg?&style=for-the-badge&logo=instagram&logoColor=white&color=black" /></a>
  <a href="https://www.linkedin.com/in/leonardo-bravo-4120b8228/"><img src="https://img.shields.io/badge/linkedin-%2312100E.svg?&style=for-the-badge&logo=linkedin&logoColor=white&color=black" /></a>
</p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>
