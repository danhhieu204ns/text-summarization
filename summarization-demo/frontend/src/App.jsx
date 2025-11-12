import { useState } from 'react'
import axios from 'axios'
import './App.css'

function App() {
  const [text, setText] = useState('')
  const [mode, setMode] = useState('both')
  const [topK, setTopK] = useState('')
  const [summary, setSummary] = useState({})
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    try {
      const payload = {
        text,
        mode
      }
      // Add top_k only if it's provided and mode includes extractive
      if (topK && (mode === 'extractive' || mode === 'both')) {
        payload.top_k = parseInt(topK)
      }
      const response = await axios.post('http://localhost:8000/summarize', payload)
      setSummary(response.data)
    } catch (error) {
      console.error(error)
      setSummary({ error: 'Error occurred' })
    }
    setLoading(false)
  }

  return (
    <div className="App">
      <h1>Text Summarization Demo</h1>
      <form onSubmit={handleSubmit}>
        <div>
          <label htmlFor="text">Input Text:</label>
          <textarea
            id="text"
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={10}
            cols={50}
            required
          />
        </div>
        <div>
          <label htmlFor="mode">Mode:</label>
          <select id="mode" value={mode} onChange={(e) => setMode(e.target.value)}>
            <option value="extractive">Extractive</option>
            <option value="abstractive">Abstractive</option>
            <option value="both">Both</option>
          </select>
        </div>
        {(mode === 'extractive' || mode === 'both') && (
          <div>
            <label htmlFor="topK">Top K (số câu trích xuất, để trống = tự động):</label>
            <input
              id="topK"
              type="number"
              value={topK}
              onChange={(e) => setTopK(e.target.value)}
              min="1"
              placeholder="Auto"
            />
          </div>
        )}
        <button type="submit" disabled={loading}>
          {loading ? 'Summarizing...' : 'Summarize'}
        </button>
      </form>
      {Object.keys(summary).length > 0 && (
        <div className="summaries-container">
          <h2>Summaries</h2>
          {summary.extractive && (
            <div className="summary-box">
              <h3>Extractive Summary</h3>
              <p>{summary.extractive}</p>
            </div>
          )}
          {summary.abstractive && (
            <div className="summary-box">
              <h3>Abstractive Summary</h3>
              <p>{summary.abstractive}</p>
            </div>
          )}
          {summary.error && (
            <div className="error-message">
              <p>{summary.error}</p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default App
