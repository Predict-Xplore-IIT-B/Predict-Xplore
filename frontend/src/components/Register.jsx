import React, { useState, useEffect } from "react";
import logo from "../assets/mlLogo.png";
import { Link, useNavigate } from "react-router-dom";
import axios from "axios";
import { useDispatch } from "react-redux";
import { addUser } from "../redux/reducers/userSlice";
import {toast, ToastContainer, Slide} from "react-toastify";
import "react-toastify/dist/ReactToastify.css";



function Register({ role }) {
  const [formData, setFormData] = useState({
    username: "",
    phone_number: "",
    email: "",
    password: "",
    confirm_password: "",
  });
  const [loading, setLoading] = useState(false);

  const navigate = useNavigate();
  const dispatch = useDispatch();

  
  // default focus on username input
  useEffect(() => {
    document.querySelector("input[name='username']").focus();
  }, []);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
  };

  const resetForm = () => {
    setFormData({
      username: "",
      phone_number: "",
      email: "",
      password: "",
      confirm_password: "",
    });
  };

  const handleRegister = async (e) => {
    e.preventDefault();
    const { username, phone_number, email, password, confirm_password } = formData;

    if (!username || !phone_number || !email || !password || !confirm_password) {
      toast.warning("Please fill out all fields.", {autoClose:3000});
      return
    }
    if (password !== confirm_password) {
      toast.warning("Passwords do not match.", {autoClose:3000});
      return 
    }


    const data = { username, phone_number, email, password, confirm_password, role };
    const user_data = { username, email, phone_number, role, password, token: "", isActive: false };

    setLoading(true);

    let response;
    try {
      response = await axios.post("http://127.0.0.1:8000/auth/register", data, {
        headers: { "Content-Type": "application/json" },
      });

      if (response.status === 201) {
        dispatch(addUser(user_data));
        resetForm();
        toast.success("Registration successful!", {autoClose: 2000, onClose:() => {navigate("/otp", { state: { view: "email" } });}});
      } else {
        console.log("try else",response.status);
        toast.error(response.data.error || "An Unexpected Error Occurred.", {autoClose: 3000});
      }
    } catch (error) {
      console.log("catch")
      if (error.response) {
        // The request was made, and the server responded with a status code
        toast.error(error.response.data.error || "An unexpected error occurred.", { autoClose: 3000 });
      } else if (error.request) {
        // The request was made but no response was received
        toast.error("No response from the server. Please try again.", { autoClose: 3000 });
      } else {
        // Something happened in setting up the request
        toast.error("An error occurred: " + error.message, { autoClose: 3000 });
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="w-screen h-screen bg-[#EAECFF] flex relative">
      <div className="w-[35%] h-full bg-white text-black rounded-[20px] flex items-center justify-center">
        <div className="w-[70%] bg-[#EAECFF] flex flex-col items-center rounded-xl px-5 py-3">
          <h1 className="text-4xl font-semibold text-[#52588D] mt-6">Register</h1>
          <form className="mt-5 w-full" onSubmit={handleRegister}>
            {["username", "phone_number", "email", "password", "confirm_password"].map((field) => (
              <input
                key={field}
                type={field.includes("password") ? "password" : "text"}
                name={field}
                placeholder={`Enter ${field.replace("_", " ").replace(/^\w/, (c) => c.toUpperCase())}`}
                className="mb-4 p-3 border border-gray-300 rounded-3xl w-full bg-white placeholder-gray-700"
                value={formData[field]}
                onChange={handleInputChange}
              />
            ))}
            <div className="flex flex-col items-center mt-4">
              <button
                type="submit"
                className={`p-3 rounded-3xl w-1/2 mb-2 ${loading ? "bg-gray-400" : "bg-[#6966FF]"}`}
                disabled={loading}
              >
                {loading ? "Registering..." : "Register"}
              </button>
              <button
                type="button"
                className="bg-white text-sky-700 p-3 rounded-3xl w-1/2 h-13 border-4 border-[#6966FF]"
                onClick={resetForm}
              >
                Cancel
              </button>
            </div>
          </form>
          <div className="mt-4 text-center text-sm text-gray-500">
            <p>
              Already have an account?{" "}
              <Link to="/login" className="text-blue-500">
                Login
              </Link>
            </p>
          </div>
        </div>
      </div>
      <div className="w-[65%] h-full flex flex-col items-center justify-center">
        <h1 className="text-[#123087] font-sans font-bold text-7xl">Predict Xplore</h1>
        <img src={logo} alt="Machine Learning Logo" className="h-[55%] w-[40%] mt-9" />
      </div>
    </main>
  );
}

export default Register;
