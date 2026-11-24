import { Component, StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

// Safety net: if any component crashes, show a message instead of a blank page.
class ErrorBoundary extends Component {
  state = { failed: false }

  static getDerivedStateFromError() {
    return { failed: true }
  }

  componentDidCatch(error) {
    console.error('UI error:', error)
  }

  render() {
    if (!this.state.failed) return this.props.children

    return (
      <div style={{ padding: '48px', color: '#e5e7eb', fontFamily: 'sans-serif' }}>
        <h2>Something went wrong while displaying this page.</h2>
        <p>Please reload the page and try again.</p>
        <button onClick={() => window.location.reload()}>Reload</button>
      </div>
    )
  }
}

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </StrictMode>,
)
