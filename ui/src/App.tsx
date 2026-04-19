import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { Toaster } from 'sonner'
import SimPage from "./pages/SimPage";
import RealPage from "./pages/RealPage";
import NotFound from "./pages/NotFound.tsx";

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-50 w-screen">
        <Toaster position="top-right" richColors />
        <Routes>
          <Route path="/" element={<SimPage />} />
          <Route path="/real" element={<RealPage />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </div>
    </Router>
  )
}

export default App
