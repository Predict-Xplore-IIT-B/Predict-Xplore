import React, { useState, useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import axios from "axios";
import logo from "../assets/mlLogo.png";
import { useDispatch } from "react-redux";
import { addLogedUser } from "../redux/reducers/userSlice";
import { toast } from "react-toastify";

function Login() {
  // New: state to hold Google user data after OAuth
  const [googleUserData, setGoogleUserData] = useState(null);
  const [username, setUsername] = useState(""); // username only
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const dispatch = useDispatch();
  const navigate = useNavigate();

  // On mount, check if redirected from Google OAuth with user data (simulate for now)
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const email = params.get("email");
    const name = params.get("name");
    if (email) {
      setGoogleUserData({ email, name });
      // Optionally, you could fetch the username from backend if you want to pre-fill
    }
  }, []);

  const handleLogin = async (e) => {
    e.preventDefault();
    if (username === "" || password === "") {
      setError("Username and password are required.");
      return;
    }
    const data = { username, password };
    try {
      setLoading(true);
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
        const token = response.data.token;
        dispatch(addLogedUser({ username, phone_number, role, roles, email: username, password, token, isActive: true }));
        setPassword("");
        setUsername("");
        toast.success("Logged In Successfully", {
          autoClose: 2000, onClose: () => {
            navigate("/otp", { state: { view: "otp" } });
          }
        });
      } else {
        toast.error(error.response.data.error || "An unexpected error occurred.", { autoClose: 3000 });
      }
    } catch (error) {
      if (error.response) {
        toast.error(error.response.data.error || "An unexpected error occurred.", { autoClose: 3000 });
      } else if (error.request) {
        toast.error("No response from the server. Please try again.", { autoClose: 3000 });
      } else {
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
        <h1 className="text-4xl font-semibold text-[#52588D] mt-6">Login</h1>
          <img src={logo} alt="Predict Xplore Logo" className="h-20 w-20 mt-6 mb-2" />
          <h1 className="text-4xl font-semibold text-[#52588D] mb-4">Predict Xplore</h1>
          {!googleUserData ? (
            <>
              <button
                className="bg-white text-black border border-gray-300 rounded-3xl w-full py-3 flex items-center justify-center mb-4 shadow hover:shadow-lg transition"
                onClick={() => {
                  window.location.href = `http://localhost:8000/auth/login/google-oauth2/?next=http://localhost:5173/login`;
                }}
              >
                <img src="https://developers.google.com/identity/images/g-logo.png" alt="Google" className="h-6 w-6 mr-2" />
                Sign in with Google
              </button>
              <div className="mt-4 text-center text-sm text-gray-500">
                Don't have an account?{" "}
                <Link to="/" className="text-blue-500">
                  Register
                </Link>
              </div>
            </>
          ) : (
            <>
              <div className="mb-2 text-center text-gray-700">
                Welcome, <span className="font-bold">{googleUserData.name || googleUserData.email}</span><br />
                Please log in to continue:
              </div>
              <form className="mt-2 w-full" onSubmit={handleLogin}>
                <input
                  type="text"
                  placeholder="Enter Username"
                  className="mb-4 p-3 border border-gray-300 rounded-3xl w-full bg-white placeholder-gray-700"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
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
                      setUsername("");
                      setPassword("");
                      setError("");
                    }}
                  >
                    Cancel
                  </button>
                </div>
              </form>
            </>
          )}
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
