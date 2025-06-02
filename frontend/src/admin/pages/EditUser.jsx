import React, { useState } from "react";
import AdminNavbar from "../../components/AdminNavbar";
import userProfile from "../../assets/userProfile.png";

const EditUser = () => {
  const [selectedRoles, setSelectedRoles] = useState(["Cse_dept", "AIML-dept"]);
  const roles = [
    "Cse_dept",
    "AIML-dept",
    "aerospace_dept",
    "cyber_dept",
    "mech_dept",
  ];

  const toggleRole = (role) => {
    if (selectedRoles.includes(role)) {
      setSelectedRoles(selectedRoles.filter((r) => r !== role));
    } else {
      setSelectedRoles([...selectedRoles, role]);
    }
  };

  return (
    <div className="bg-[#EAECFF] min-h-screen">
      {/* Navbar */}
      <div className="">
        <AdminNavbar />
      </div>

      {/* Content */}
      <div className="container mx-auto p-6">
        {/* Heading */}
        <h1 className="text-3xl font-semibold text-center text-[#39407D] mb-4">
          Edit User
        </h1>

        {/* Search Bar */}
        <div className="flex justify-center mb-4">
          <div className="relative w-4/5">
            <input
              type="text"
              placeholder="Firstname Secondname"
              className="w-full border border-gray-300 rounded-full px-4 py-2 text-[#39407D] focus:outline-none focus:ring-2 focus:ring-[#6966FF] pr-10"
            />
            <span className="absolute inset-y-0 right-3 flex items-center">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
                strokeWidth="2"
                stroke="currentColor"
                className="w-5 h-5 text-gray-400"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M21 21l-4.35-4.35m0 0a7.5 7.5 0 1 0-10.607-10.607 7.5 7.5 0 0 0 10.607 10.607z"
                />
              </svg>
            </span>
          </div>
        </div>

        {/* User Details and Roles Section */}
        <div className="flex flex-col lg:flex-row justify-center items-center bg-white rounded-lg shadow-lg h-[458px] p-6 gap-8 w-4/5 mx-auto">
          {/* Left Section (User Info) */}
          <div className="w-full lg:w-1/2">
            {/* Profile Picture and Details */}
            <div className="flex items-center mb-6">
              <img
                src={userProfile}
                alt="User"
                className="w-20 h-20 rounded-full shadow-lg"
              />
              <div className="ml-4">
                <h2 className="text-xl font-medium text-[#39407D]">firstname</h2>
                <p className="text-gray-600">yourname@gmail.com</p>
              </div>
            </div>

            {/* User Details */}
            <div className="space-y-4">
              <div>
                <label
                  htmlFor="name"
                  className="block text-gray-600 mb-1 font-medium"
                >
                  Name
                </label>
                <input
                  id="name"
                  type="text"
                  placeholder="fullname"
                  className="w-full border border-gray-300 rounded-full px-4 py-2 focus:outline-none focus:ring-2 focus:ring-[#6966FF]"
                />
              </div>

              <div>
                <label
                  htmlFor="email"
                  className="block text-gray-600 mb-1 font-medium"
                >
                  Email account
                </label>
                <input
                  id="email"
                  type="email"
                  placeholder="yourname@gmail.com"
                  className="w-full border border-gray-300 rounded-full px-4 py-2 focus:outline-none focus:ring-2 focus:ring-[#6966FF]"
                />
              </div>

              <div>
                <label
                  htmlFor="mobile"
                  className="block text-gray-600 mb-1 font-medium"
                >
                  Mobile number
                </label>
                <input
                  id="mobile"
                  type="text"
                  placeholder="Add number"
                  className="w-full border border-gray-300 rounded-full px-4 py-2 focus:outline-none focus:ring-2 focus:ring-[#6966FF]"
                />
              </div>

              <div>
                <label
                  htmlFor="location"
                  className="block text-gray-600 mb-1 font-medium"
                >
                  Location
                </label>
                <input
                  id="location"
                  type="text"
                  placeholder="USA"
                  className="w-full border border-gray-300 rounded-full px-4 py-2 focus:outline-none focus:ring-2 focus:ring-[#6966FF]"
                />
              </div>
            </div>
          </div>

          {/* Right Section (User Roles) */}
          <div className="w-full lg:w-1/2">
            <h3 className="text-xl font-medium text-[#39407D] mb-4">User Roles</h3>
            <div className="space-y-3">
              {roles.map((role) => (
                <div
                  key={role}
                  className="flex items-center justify-between bg-gray-100 rounded-lg p-3 shadow-sm"
                >
                  <span className="text-gray-700">{role}</span>
                  <div
                    className={`w-6 h-6 flex items-center justify-center rounded-full cursor-pointer ${
                      selectedRoles.includes(role)
                        ? "bg-green-500"
                        : "bg-gray-300"
                    }`}
                    onClick={() => toggleRole(role)}
                  >
                    {selectedRoles.includes(role) && (
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="white"
                        strokeWidth="3"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        className="w-4 h-4"
                      >
                        <polyline points="20 6 9 17 4 12" />
                      </svg>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Save Button */}
        <div className="mt-4 flex justify-center">
          <button className="bg-[#6966FF] text-white px-6 py-2 rounded-full shadow-lg text-lg">
            Save Changes
          </button>
        </div>
      </div>
    </div>
  );
};

export default EditUser;
