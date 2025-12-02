"use client"

import { useEffect, useRef } from "react"
import type { BufferAttribute } from "three"

export default function ThreeBackground() {
  const containerRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    if (typeof window === "undefined") return

    let cleanup: (() => void) | null = null

    const init = async () => {
      const container = containerRef.current
      if (!container) return

      const THREE = await import("three")

      const sizes = {
        width: container.clientWidth || window.innerWidth,
        height: container.clientHeight || window.innerHeight,
      }

      const scene = new THREE.Scene()
      scene.fog = new THREE.FogExp2(0xd6e9ff, 0.04)

      const camera = new THREE.PerspectiveCamera(45, sizes.width / sizes.height, 0.1, 100)
      camera.position.set(0, 1.4, 9)

      const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
      renderer.setSize(sizes.width, sizes.height)
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
      renderer.setClearColor(0xeaf4ff, 1)
      container.appendChild(renderer.domElement)

      const ambientLight = new THREE.AmbientLight(0xbcd5ff, 0.9)
      scene.add(ambientLight)

      const topLight = new THREE.PointLight(0x88b4ff, 1.3)
      topLight.position.set(-4, 6, 4)
      scene.add(topLight)

      const rimLight = new THREE.PointLight(0x7cf0ff, 1.0)
      rimLight.position.set(6, -3, -2)
      scene.add(rimLight)

      // Globo low-poly con más matices
      const globeGeometry = new THREE.IcosahedronGeometry(1.7, 2)
      const globeMaterial = new THREE.MeshStandardMaterial({
        color: 0x88a9ff,
        metalness: 0.45,
        roughness: 0.3,
        emissive: 0x4765ff,
        emissiveIntensity: 0.18,
        flatShading: true,
      })
      const globe = new THREE.Mesh(globeGeometry, globeMaterial)
      globe.position.set(0, -0.1, -1.1)
      scene.add(globe)

      // Superficie suave tipo niebla debajo del texto
      const geometry = new THREE.PlaneGeometry(22, 10, 160, 50)
      const material = new THREE.MeshStandardMaterial({
        color: 0xe4ecff,
        metalness: 0.15,
        roughness: 0.85,
        side: THREE.DoubleSide,
      })

      const wave = new THREE.Mesh(geometry, material)
      wave.rotation.x = -Math.PI / 2.4
      wave.position.y = -1.25
      scene.add(wave)

      const positionAttribute = geometry.getAttribute("position") as BufferAttribute
      const initialPositions = (positionAttribute.array as Float32Array).slice() as Float32Array

      // "Pájaros" (aviones de papel) orbitando el globo
      const planeGeometry = new THREE.ConeGeometry(0.14, 0.5, 3)
      const planeMaterial = new THREE.MeshStandardMaterial({
        color: 0xf9fbff,
        roughness: 0.35,
        metalness: 0.1,
        side: THREE.DoubleSide,
      })

      const planes: any[] = []
      const planeConfigs: { radius: number; speed: number; offset: number; height: number }[] = []

      const planeCount = 36
      for (let i = 0; i < planeCount; i++) {
        const mesh = new THREE.Mesh(planeGeometry, planeMaterial)
        const radius = 3.2 + Math.random() * 1.4
        const speed = 0.14 + Math.random() * 0.2
        const offset = Math.random() * Math.PI * 2
        const height = (Math.random() - 0.2) * 1.0
        planeConfigs.push({ radius, speed, offset, height })
        planes.push(mesh)
        scene.add(mesh)
      }

      // Nubes suaves en el cielo
      const cloudGeometry = new THREE.SphereGeometry(0.55, 16, 16)
      const cloudMaterial = new THREE.MeshStandardMaterial({
        color: 0xffffff,
        roughness: 0.95,
        metalness: 0,
        transparent: true,
        opacity: 0.9,
      })

      const clouds: any[] = []
      const cloudConfigs: { baseX: number; baseY: number; baseZ: number; speed: number; offset: number }[] = []

      const cloudCount = 8
      for (let i = 0; i < cloudCount; i++) {
        const mesh = new THREE.Mesh(cloudGeometry, cloudMaterial)
        const baseX = -6 + Math.random() * 12
        const baseY = 1.3 + Math.random() * 0.6
        const baseZ = -4 - Math.random() * 4
        const speed = 0.02 + Math.random() * 0.03
        const offset = Math.random() * Math.PI * 2
        cloudConfigs.push({ baseX, baseY, baseZ, speed, offset })
        clouds.push(mesh)
        scene.add(mesh)
      }

      const clock = new THREE.Clock()
      let frameId: number

      const animate = () => {
        const t = clock.getElapsedTime()

        for (let i = 0; i < positionAttribute.count; i++) {
          const ix = i * 3
          const x = initialPositions[ix]
          const y = initialPositions[ix + 1]

          const base = Math.sin(x * 0.45 + t * 0.7) * 0.08
          const cross = Math.cos((x + y) * 0.32 - t * 0.45) * 0.05
          const ripple = Math.sin(x * 1.1 - y * 0.7 + t * 1.1) * 0.03

          const z = base + cross + ripple
          positionAttribute.setZ(i, z)
        }
        positionAttribute.needsUpdate = true

        // Rotación suave del globo y olas mínimas debajo
        globe.rotation.y = t * 0.16
        wave.rotation.z = Math.sin(t * 0.12) * 0.06

        // Actualizar "pájaros" orbitando
        for (let i = 0; i < planes.length; i++) {
          const cfg = planeConfigs[i]
          const angle = cfg.offset + t * cfg.speed
          const radius = cfg.radius
          const x = Math.cos(angle) * radius
          const z = Math.sin(angle) * radius
          const y = cfg.height + Math.sin(t * 0.7 + cfg.offset) * 0.18

          const plane = planes[i]
          plane.position.set(x, y, z)
          plane.lookAt(globe.position)
          plane.rotation.x = Math.PI / 2.4
          plane.rotation.z += 0.015
        }

        // Actualizar nubes
        for (let i = 0; i < clouds.length; i++) {
          const cfg = cloudConfigs[i]
          const x = cfg.baseX + Math.sin(t * cfg.speed + cfg.offset) * 1.4
          const y = cfg.baseY + Math.sin(t * 0.35 + cfg.offset) * 0.12
          const z = cfg.baseZ
          const cloud = clouds[i]
          cloud.position.set(x, y, z)
        }

        renderer.render(scene, camera)
        frameId = requestAnimationFrame(animate)
      }

      animate()

      const handleResize = () => {
        const width = container.clientWidth || window.innerWidth
        const height = container.clientHeight || window.innerHeight

        camera.aspect = width / height
        camera.updateProjectionMatrix()
        renderer.setSize(width, height)
      }

      window.addEventListener("resize", handleResize)

      cleanup = () => {
        window.removeEventListener("resize", handleResize)
        if (frameId) cancelAnimationFrame(frameId)
        geometry.dispose()
        material.dispose()
        planeGeometry.dispose()
        planeMaterial.dispose()
        cloudGeometry.dispose()
        cloudMaterial.dispose()
        globeGeometry.dispose()
        globeMaterial.dispose()
        renderer.dispose()
        if (container.contains(renderer.domElement)) {
          container.removeChild(renderer.domElement)
        }
      }
    }

    init().catch(() => {
      // Si three falla por cualquier motivo, no rompemos la UI
    })

    return () => {
      if (cleanup) cleanup()
    }
  }, [])

  return <div ref={containerRef} className="absolute inset-0" style={{ pointerEvents: "none", zIndex: 1 }} />
}
