import React from 'react'
import '../styles/dashboard.css'
import { Link } from 'react-router-dom';


const Dashboard = () => {
  return (
    <div className="dashboard ">
      <h1 className="dashboard-title">Admin Dashboard</h1>

      <div className="stats-container">
        <div className="stat-box">
          <p>Total Users</p>
          <h2>2376</h2>
        </div>
        <div className="stat-box">
          <p>Active Users</p>
          <h2>3645</h2>
        </div>
        <div className="stat-box">
          <p>Active Models</p>
          <h2>5645</h2>
        </div>
      </div>

      <div className="actions-container">
        <Link to="/admin/manage-user">
          <div className="action-box" >
            <i className="icon">👤</i>
            <p>Manage Users</p>
          </div>
        </Link>
        <Link to="/admin/create-model">
          <div className="action-box">
            <i className="icon">➕</i>
            <p>Create Model</p>
          </div>
        </Link>
        <Link to="/admin/create-pipeline">
          <div className="action-box">
            <i className="icon">➕</i>
            <p>Create Pipeline</p>
          </div>
        </Link>
        <Link to="/admin/model-test">
          <div className="action-box">
            <i className="icon">⚙️</i>
            <p>Test Model</p>
          </div>
        </Link>
      </div>
    </div>
  );
};

export default Dashboard;

