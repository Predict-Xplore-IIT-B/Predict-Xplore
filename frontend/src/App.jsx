import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import './App.css';
import Homepage from './user/Homepage';
import Modeltestpage from './user/Modeltestpage';
import Modeltestrunpage from './user/Modeltestrunpage';
import Reportpage from './user/Reportpage';
import Login from './components/Login';
import OTP from './components/OTP';
import Register from './components/Register';
import CreateModel from './admin/pages/CreateModel';
import AdminModelTest from './admin/pages/AdminModelTest';
import CreatePipeline from './admin/pages/CreatePipeline';
import ModelProceed from './admin/pages/ModelProceed';
import AdminDashboard from './admin/pages/AdminDashboard';
import AdminReport from './admin/pages/AdminReport';
import AdminModelList from './admin/pages/AdminModelList';
import { ToastContainer,Slide } from 'react-toastify';
import ManageUser from './admin/pages/ManageUser';

function App() {
  return (
    <div>
      <ToastContainer position="top-left" transition={Slide} className="mt-10" />
      <Router>
        <Routes>
          <Route path='/' element={<Register role="user"/> }/>
          {/* login part */}
          <Route path="/login" element={<Login />} />
          <Route path="/otp" element={<OTP />} />
          <Route path="/user/register" element={<Register role="user" />} />
          <Route path="/admin/register" element={<Register role="admin" />} />
          {/* login part end */}
          <Route path="/home" element={<Homepage />} />
          <Route path="/model-test" element={<Modeltestpage />} />
          <Route path="/model-test-run" element={<Modeltestrunpage />} />
          <Route path="/reports" element={<Reportpage />} />

          {/* Admin Pages */}
          <Route path="/admin/dashboard" element={<AdminDashboard />} />
          <Route path="/admin/models" element={<AdminModelList />} />
          <Route path="/admin/create-model" element={<CreateModel />} />
          <Route path="/admin/create-pipeline" element={<CreatePipeline />} />  
          <Route path="/admin/reports" element={<AdminReport />} />
          
          <Route path="/admin/manage-user" element={<ManageUser />} />
          <Route path="/admin/model-proceed" element={<ModelProceed />} />
          <Route path="/admin/model-test" element={<AdminModelTest />} />
        </Routes>
      </Router>
    </div>
  );
}

export default App;