import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import axios from "axios"; // Import Axios
import logo from "../assets/mlLogo.png";
import { useDispatch } from "react-redux";
import { addLogedUser } from "../redux/reducers/userSlice";
import { toast } from "react-toastify";

function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const dispatch = useDispatch();
  const navigate = useNavigate();



  const handleLogin = async (e) => {
    e.preventDefault(); // Prevent default form submission

    if (email === "" || password === "") {
      setError("Email and Password in Mandatory");
      return
    }
    const data = { email, password };

    try {
      setLoading(true);
      console.log(data)
      const response = await axios.post("http://127.0.0.1:8000/auth/login", data, {
        headers: {
          "Content-Type": "application/json",
        },
      });
      if (response.status === 200) {
        const username = response.data.username;
        const phone_number = response.data.phone_number;
        const role = response.data.role;
        const roles = response.data.roles;
        const token = response.data.token; // Assuming the backend returns a token
        dispatch(addLogedUser({ username, phone_number,role, roles, email, password, token, isActive: true })); // Update user in Redux store
        console.log(response.data); // Log the response from Django backend
        setPassword(""); // Clear password after successful login
        setEmail("");
        toast.success("Logged In Successfuly", {
          autoClose: 2000, onClose: () => {
            navigate("/otp", { state: { view: "otp" } });
          }
        })
      }
      else {
        toast.error(error.response.data.error || "An unexpected error occurred.", { autoClose: 3000 });
      }
    } catch (error) {
      console.error(error);
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
    }
    finally {
      setLoading(false);
    }
  };



  return (
    <main className="w-screen h-screen bg-[#EAECFF] flex relative">
      <div className="w-[35%] h-full bg-white text-black rounded-[20px] flex items-center justify-center">
        <div className="w-[70%] bg-[#EAECFF] flex flex-col items-center rounded-xl px-5 py-3">
          <h1 className="text-4xl font-semibold text-[#52588D] mt-6">Login</h1>
          <form className="mt-5 w-full" onSubmit={handleLogin}>
            <input
              type="text"
              placeholder="Enter Email"
              className="mb-4 p-3 border border-gray-300 rounded-3xl w-full bg-white placeholder-gray-700"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
            <input
              type="password"
              placeholder="Enter Password"
              className="mb-4 p-3 border border-gray-300 rounded-3xl w-full bg-white placeholder-gray-700"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
            {error && <p className="text-red-500 text-sm">{error}</p>}
            <div className="flex flex-col items-center mt-4">
              <button
                type="submit"
                className={`text-white p-3 rounded-3xl w-1/2 mb-2 ${loading ? "bg-gray-400" : "bg-[#6966FF]"}`}
                disabled={loading}
              >
                {loading ? "Confirming Credentials..." : "Login"}
              </button>
              <button
                type="button"
                className="bg-white text-sky-700 p-3 rounded-3xl w-1/2 h-13 border-4 border-[#6966FF]"
                onClick={() => {
                  setEmail("");
                  setPassword("");
                  setError("");
                }}
              >
                Cancel
              </button>
            </div>
          </form>
          <div className="mt-4 text-center text-sm text-gray-500">
            <p>
              Don't have an account?{" "}
              <Link to="/" className="text-blue-500">
                Register
              </Link>
            </p>
          </div>
        </div>
      </div>
      <div className="w-[65%] h-full flex flex-col items-center justify-center">
        <h1 className="text-[#123087] font-sans font-bold text-7xl">
          Predict Xplore
        </h1>
        <img src={logo} alt="Machine Learning Logo" className="h-[55%] w-[40%] mt-9" />
      </div>
    </main>
  );
}

export default Login;
