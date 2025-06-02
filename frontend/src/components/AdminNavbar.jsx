import React from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import logo from "../assets/mlLogo.png";
import { useDispatch, useSelector } from "react-redux";
import axios from "axios";
import { removeUser } from "../redux/reducers/userSlice";
import { clearModelList } from "../redux/reducers/modelSlice";

const AdminNavbar = () => {
  const location = useLocation();

  const navigate = useNavigate();
  const dispatch = useDispatch();
  const user = useSelector((state) => state.user.users[state.user.users.length -1]);
  

  const handleLogout = async () => {
  const token = user.token;
  const username = user.username;
    try{
      console.log(token)
      const response = await axios.post("http://127.0.0.1:8000/auth/logout", null, {
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Token ${token}`
        }
      })

      console.log(response.data)

      if (response.status === 200){
        dispatch(removeUser(username));
        dispatch(clearModelList());
        navigate("/login");
      }
    }
    catch(err){
      console.log(err)
      navigate("/login");
    }
  };

  // Function to dynamically apply classes for active and hover states
  const getLinkClass = (path) => {
    const baseClasses = "text-[#39407D] text-xl h-full flex items-center relative";
    const activeClasses =
      "after:content-[''] after:absolute after:bottom-0 after:left-0 after:w-full after:h-0.5 after:bg-purple-600";
    const hoverClasses = "hover:text-purple-600 transition-colors";

    return location.pathname === path
      ? `${baseClasses} ${activeClasses}`
      : `${baseClasses} ${hoverClasses}`;
  };

  return (
    <nav className="bg-[#EAECFF] h-20 p-5 flex justify-between items-center font-sans">
      {/* Logo Section */}
      <div className="flex items-center">
        <Link to="/admin/dashboard">
          <img src={logo} alt="Admin Logo" className="h-12 w-12 mr-4" />
        </Link>
      </div>

      {/* Admin Navigation Links */}
      <div className="flex items-center space-x-8">
        <Link to="/admin/dashboard" className={getLinkClass("/admin/dashboard")}>
          Dashboard
        </Link>
        <Link to="/admin/model-list" className={getLinkClass("/admin/models")}>
          Models
        </Link>
        <Link to="/admin/create-model" className={getLinkClass("/admin/create-model")}>
          Create Model
        </Link>
        <Link to="/admin/create-pipeline" className={getLinkClass("/admin/create-pipeline")}>
          Create Pipeline
        </Link>
        <Link to="/admin/reports" className={getLinkClass("/admin/reports")}>
          Reports
        </Link>
        <button
          onClick={handleLogout}
          className="text-[#39407D] text-xl font-bold h-full flex items-center cursor-pointer hover:text-purple-600 transition-colors"
        >
          Logout
        </button>
      </div>
    </nav>
  );
};

export default AdminNavbar;
