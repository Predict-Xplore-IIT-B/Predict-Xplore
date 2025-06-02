import React, { useState, useEffect, useRef } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import axios from "axios";
import { useSelector, useDispatch } from "react-redux"; // Redux hooks
import logo from "../assets/mlLogo.png";
import { updateUserStatus } from "../redux/reducers/userSlice";
import { toast } from "react-toastify";

function OTP() {
  const [otp, setOtp] = useState(Array(5).fill(""));
  const [timeLeft, setTimeLeft] = useState(300);
  const [showTimer, setShowTimer] = useState(true);
  const [loading, setLoading] = useState(false);
  const inputRefs = useRef([]);

  const location = useLocation();
  const navigate = useNavigate();
  const dispatch = useDispatch();

  const user = useSelector((state) => state.user.users[state.user.users.length - 1]);
  const email = user.email;
  const username = user.username;
  const password = user.password;

  const view = location.state.view;

  const handleOTP = async () => {
    setLoading(true);
    try {
      console.log({ otp, email });
      let response;

      if (view === "email") {
        response = await axios.post(
          "http://127.0.0.1:8000/auth/verify-email",
          { otp: otp.join(""), email },
          {
            headers: { "Content-Type": "application/json" },
          }
        );
      } else if (view === "otp") {
        console.log(user.token);
        response = await axios.post(
          "http://127.0.0.1:8000/auth/verify-otp",
          { otp: otp.join(""), email },
          {
            headers: { "Content-Type": "application/json", Authorization: "Token " + user.token },
          }
        );
      }

      if (response.status === 200) {
        toast.success("OTP verified successfully", {
          autoClose: 2000,
          onClose: () => {
            if (view === "email") {
              
              navigate("/ldpa", { state: { view: "email" } })
            } else{
              if (user.role === "admin") {
                navigate("/admin/dashboard");
              }
              else {
                navigate("/home");
              }
            }

          },
        });
      }
    } catch (error) {
      if (error.response) {
        toast.error(
          error.response.data.error || error.response.data.message || "An unexpected error occurred.",
          { autoClose: 3000 }
        );
      } else {
        toast.error("An error occurred: " + error.message, { autoClose: 3000 });
      }
    } finally {
      setLoading(false);
    }
  };

  const handleResend = async () => {
    try {
      setOtp(Array(5).fill(""));
      setTimeLeft(300);
      setShowTimer(true);

      await axios.post(
        "http://127.0.0.1:8000/auth/resend-otp",
        { username, email },
        {
          headers: { "Content-Type": "application/json" },
        }
      );
      alert("OTP resent successfully");
    } catch (error) {
      console.error(error);
      alert("Failed to resend OTP");
    }
  };

  const handleChange = (e, index) => {
    const value = e.target.value;
    if (/^\d$/.test(value) || value === "") {
      const newOtp = [...otp];
      newOtp[index] = value;
      setOtp(newOtp);

      // Move to next input if input is valid and not last input
      if (value !== "" && index < otp.length - 1) {
        inputRefs.current[index + 1]?.focus();
      }
    }
  };

  const handleKeyDown = (e, index) => {
    if (e.key === "Backspace") {
      if (otp[index] === "" && index > 0) {
        // Move to previous input if empty
        inputRefs.current[index - 1]?.focus();
      }
    }
  };

  useEffect(() => {
    if (timeLeft === 0) {
      setShowTimer(false);
      return;
    }
    const timer = setInterval(() => {
      setTimeLeft((prev) => prev - 1);
    }, 1000);

    return () => clearInterval(timer);
  }, [timeLeft]);

  const formatTime = (time) => {
    const minutes = Math.floor(time / 60);
    const seconds = time % 60;
    return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
  };

  return (
    <main className="w-screen h-screen bg-[#EAECFF] flex relative">
      <div className="w-[35%] h-full bg-white text-black rounded-[20px] flex flex-col items-center justify-center">
        <div className="w-[70%] bg-[#EAECFF] flex flex-col items-center rounded-xl px-5 py-3">
          <h1 className="text-4xl font-semibold text-[#52588D] mt-6">{view} Verification</h1>
          <form className="mt-5 w-full flex flex-col items-center">
            <div className="flex mb-4 space-x-2">
              {otp.map((digit, index) => (
                <input
                  key={index}
                  type="text"
                  maxLength="1"
                  className="w-12 h-12 text-center text-2xl border border-gray-300 rounded-lg bg-white"
                  value={digit}
                  onChange={(e) => handleChange(e, index)}
                  onKeyDown={(e) => handleKeyDown(e, index)}
                  ref={(el) => (inputRefs.current[index] = el)}
                />
              ))}
            </div>
            {showTimer && <div className="text-gray-500 mb-4">Resend in {formatTime(timeLeft)}</div>}
            <div className="flex flex-col items-center mt-4">
              <button
                type="button"
                className={`text-white p-3 rounded-3xl w-20 mb-2 ${loading ? "bg-gray-400" : "bg-[#6966FF]"}`}
                onClick={handleOTP}
                disabled={loading}
              >
                {loading ? "Verifying..." : "Submit"}
              </button>
              <button
                type="button"
                className="bg-white text-sky-700 p-3 rounded-3xl w-20 h-13 border-4 border-[#6966FF]"
                onClick={() => navigate(-1)}
              >
                Cancel
              </button>
            </div>
          </form>
          <div className="mt-4 text-center text-sm text-gray-500">
            {!showTimer && (
              <p onClick={handleResend} className="cursor-pointer">
                Didn't receive the OTP?{" "}
                <span className="text-blue-500">Resend OTP</span>
              </p>
            )}
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

export default OTP;
