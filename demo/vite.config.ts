import { defineConfig } from 'vite'

export default defineConfig({
  base: '/splathash/',
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  }
})
