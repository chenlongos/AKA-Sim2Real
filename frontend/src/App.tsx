import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom'
import Home from './pages/Home'
import About from './pages/About'
import NotFound from './pages/NotFound'

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-50 w-screen">
        <nav className="bg-white shadow-md">
          <div className="w-full px-4 py-4">
            <div className="flex space-x-6">
              <Link to="/" className="text-blue-600 hover:text-blue-800 font-medium">
                Home
              </Link>
              <Link to="/about" className="text-blue-600 hover:text-blue-800 font-medium">
                About
              </Link>
            </div>
          </div>
        </nav>

        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/about" element={<About />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </div>
    </Router>
  )
}

export default App
