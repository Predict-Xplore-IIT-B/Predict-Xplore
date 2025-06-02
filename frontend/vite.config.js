import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',  // Bind to all network interfaces
    port: 5173,
    strictPort: true,
    watch: {
      usePolling: true,  // Enable polling for hot reloading
    },
  },
})
