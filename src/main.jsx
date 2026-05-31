import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'

// Polyfill window.storage (normally provided by claude.ai) using localStorage.
// This gives the local build full persistence between browser sessions.
if (!window.storage) {
  window.storage = {
    async get(key) {
      const value = localStorage.getItem(key)
      if (value === null) throw new Error(`Key not found: ${key}`)
      return { key, value }
    },
    async set(key, value) {
      localStorage.setItem(key, value)
      return { key, value }
    },
    async delete(key) {
      localStorage.removeItem(key)
      return { key, deleted: true }
    },
    async list(prefix = '') {
      const keys = Object.keys(localStorage).filter(k => k.startsWith(prefix))
      return { keys, prefix }
    },
  }
}

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)
