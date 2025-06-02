import logo from "../assets/mlLogo.png";
import React, { useState } from "react";
import { Link, useNavigate, useLocation } from "react-router-dom"; 
import axios from "axios";  // Import Axios

function Ldap() {
  const [ldap_uid, setLdap_uid] = useState("");
  const [ldap_password, setLdap_password] = useState("");
  const [errorMessage, setErrorMessage] = useState("");

  const navigate = useNavigate();
  const location = useLocation();
  
  const view = location.state.view;


  // Handle form submission
  const handleLDPA = async (e) => {
    e.preventDefault();  // Preventing page refresh on form submission
    
    const data = { ldap_uid, ldap_password };
    console.log(data);  // We can check this data in the console to ensure it’s correct
    
    // try {
    //   const response = await axios.post('http://localhost:8000/api/ldap-login/', data);

    //   // Handlinlg the response from the backend
    //   if (response.data.status === 'success') {
    //     console.log("Login successful");
    //     // Redirecting or showing success message
    //   } else {
    //     setErrorMessage(response.data.message || 'Invalid credentials');
    //   }
    // } catch (error) {
    //   console.error("Error logging in:", error);
    //   setErrorMessage("Error connecting to server");
    // }
    if (view === "email" ){
      navigate("/admin/dashboard");
    }
    else{
      navigate("/otp", { state: { view: "otp" } });
    }
  };

  return (
    <main className="w-screen h-screen bg-[#EAECFF] flex relative">
      <div className="w-[35%] h-full bg-white text-black rounded-[20px] flex items-center justify-center">
        <div className="w-[70%] bg-[#EAECFF] flex flex-col items-center rounded-xl px-5 py-3">
          <h1 className="text-4xl font-semibold text-[#52588D] mt-6">LDAP Login</h1>
          <form className="mt-5 w-full" onSubmit={handleLDPA}>
            <input
              type="text"
              placeholder="Enter LDAP Username"
              className="mb-4 p-3 border border-gray-300 rounded-3xl w-full bg-white placeholder-gray-700"
              value={ldap_uid}
              onChange={(e) => setLdap_uid(e.target.value)}
            />
            <input
              type="password"
              placeholder="Enter LDAP Password"
              className="mb-4 p-3 border border-gray-300 rounded-3xl w-full bg-white placeholder-gray-700"
              value={ldap_password}
              onChange={(e) => setLdap_password(e.target.value)}
            />
            <div className="flex flex-col items-center mt-4">
              <button
                type="submit"
                className="bg-[#6966FF] text-white p-3 rounded-3xl w-1/2 mb-2"
              >
                Login
              </button>
              <button
                type="button"
                className="bg-white text-sky-700 p-3 rounded-3xl w-1/2 h-13 border-4 border-[#6966FF]"
              >
                Cancel
              </button>
            </div>
          </form>

          {errorMessage && <p style={{ color: 'red', marginTop: '10px' }}>{errorMessage}</p>}

          <div className="mt-4 text-center text-sm text-gray-500">
            <p>
              Need help? <Link to="/support" className="text-blue-500">Support</Link> 
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

export default Ldap;
