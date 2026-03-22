---
name: User profile
description: Who the user is, their background and working style
type: user
---

- Building a production GPU-hosted AI proctoring SaaS (ProctorPod) on AWS EC2 g4dn.xlarge
- Comfortable with Python backend (FastAPI, aiortc, YOLO, MediaPipe), Docker, AWS, Next.js
- Wants production-grade code — no dead code, no CPU fallbacks, no placeholder hacks
- Deploys by rebuilding Docker image, pushing to DockerHub, pulling fresh on EC2
- Works across two Docker containers (proctor1:8000, proctor2:8001) on the same EC2 instance
- Frontend env vars: `NEXT_PUBLIC_BACKEND_URL`, `NEXT_PUBLIC_BACKEND_URL_2`
