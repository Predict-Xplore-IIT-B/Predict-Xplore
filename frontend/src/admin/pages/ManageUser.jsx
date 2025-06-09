import React, { useState } from "react";
import { MdEdit, MdDelete } from "react-icons/md";
import AdminNavbar from "../../components/AdminNavbar";
import userProfile from "../../assets/userProfile.png";

const rolesList = [
  "Cse_dept",
  "AIML_dept",
  "aerospace_dept",
  "cyber_dept",
  "mech_dept",
];

const ManageUser = () => {
  const [users, setUsers] = useState([
    {
      id: 1,
      name: "john doe",
      phone_number: "9876543210",
      email: "john@example.com",
      roles: ["Cse_dept", "AIML-dept"],
      location: "New York",
    },
    {
      id: 2,
      name: "jane smith",
      phone_number: "9123456780",
      email: "jane@example.com",
      roles: ["cyber_dept"],
      location: "California",
    },
  ]);

  const [modalVisible, setModalVisible] = useState(false);
  const [isEditMode, setIsEditMode] = useState(false);

  const [formData, setFormData] = useState({
    id: null,
    name: "",
    phone_number: "",
    email: "",
    roles: [],
    location: "",
  });

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const toggleRole = (role) => {
    if (formData.roles.includes(role)) {
      setFormData((prev) => ({
        ...prev,
        roles: prev.roles.filter((r) => r !== role),
      }));
    } else {
      setFormData((prev) => ({
        ...prev,
        roles: [...prev.roles, role],
      }));
    }
  };

  const handleRowClick = (user) => {
    setFormData(user);
    setIsEditMode(false);
    setModalVisible(true);
  };

  const handleEditUser = (user) => {
    setFormData(user);
    setIsEditMode(true);
    setModalVisible(true);
  };

  const handleAddUser = () => {
    setFormData({
      id: null,
      name: "",
      phone_number: "",
      email: "",
      roles: [],
      location: "",
    });
    setIsEditMode(true);
    setModalVisible(true);
  };

  const handleDeleteUser = (id) => {
    setUsers(users.filter((user) => user.id !== id));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (formData.id) {
      setUsers(users.map((u) => (u.id === formData.id ? formData : u)));
    } else {
      const newUser = { ...formData, id: Date.now() };
      setUsers([...users, newUser]);
    }
    setModalVisible(false);
  };

  return (
    <>
      <AdminNavbar />
      <div className="bg-[#EAECFF] min-h-screen">
        <div className="container mx-auto px-6 pt-8">
          <div className="user-list-header flex justify-between items-center mb-6">
            <h2 className="text-3xl font-semibold text-[#39407D]">Manage Users</h2>
            <button
              className="bg-[#6966ff] text-white px-4 py-2 rounded-full shadow-lg"
              onClick={handleAddUser}
            >
              ➕ Add User
            </button>
          </div>

          <div className="grid grid-cols-5 font-semibold text-[#39407D] mb-3  ml-4">
            <div>Name</div>
            <div>Email</div>
            <div>Phone</div> 
            <div>Roles</div>
            <div>Actions</div>
          </div>

          {users.map((user) => (
            <div
              key={user.id}
              className="grid grid-cols-5 bg-white p-4 border border-[#9f9df1] rounded-xl mb-2 shadow cursor-pointer hover:bg-gray-100"
              onClick={() => handleRowClick(user)}
            >
              <div>{user.name}</div>
              <div>{user.email}</div>
              <div>{user.phone_number}</div>  
              <div>{user.roles.join(", ")}</div>
              <div className="flex space-x-2" onClick={(e) => e.stopPropagation()}>
                <button onClick={() => handleEditUser(user)}>
                  <MdEdit className="text-blue-500 text-xl ml-4" />
                </button>
                <button onClick={() => handleDeleteUser(user.id)}>
                  <MdDelete className="text-red-500 text-xl" />
                </button>
              </div>
            </div>
          ))}
        </div>

        {/* Modal */}
        {modalVisible && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center z-50">
            <form
              onSubmit={handleSubmit}
              className="bg-white rounded-lg w-[90%] lg:w-[70%] p-6 shadow-xl"
            >
              <h1 className="text-2xl font-bold text-center text-[#39407D] mb-6">
                {formData.id
                  ? isEditMode
                    ? "Edit User"
                    : "User Details"
                  : "Add User"}
              </h1>

              <div className="flex flex-col lg:flex-row gap-8">
                {/* Left Section */}
                <div className="lg:w-1/2 space-y-4">
                  <div className="flex items-center gap-4">
                    <img
                      src={userProfile}
                      alt="Profile"
                      className="w-16 h-16 rounded-full shadow-md"
                    />
                    <div>
                      <h2 className="text-lg font-semibold text-[#39407D]">
                        {formData.name||"firstname"}
                      </h2>
                      <p className="text-gray-600">{formData.email || "yourname@gmail.com"}</p>
                    </div>
                  </div>

                  {/* Name Field */}
                  <div>
                    <label className="block text-gray-600 font-medium mb-1">Name</label>
                    <input
                      type="text"
                      name="name"
                      value={formData.name}
                      onChange={handleInputChange}
                      placeholder="fullname"
                      readOnly={!isEditMode}
                      className="w-full border border-gray-300 rounded-full px-4 py-2 focus:outline-none focus:ring-2 focus:ring-[#6966FF]"
                    />
                  </div>

                  {/* Email Field */}
                  <div>
                    <label className="block text-gray-600 font-medium mb-1">Email Account</label>
                    <input
                      type="email"
                      name="email"
                      value={formData.email}
                      onChange={handleInputChange}
                      placeholder="yourname@gmail.com"
                      readOnly={!isEditMode}
                      className="w-full border border-gray-300 rounded-full px-4 py-2 focus:outline-none focus:ring-2 focus:ring-[#6966FF]"
                    />
                  </div>

                  {/* Mobile Number Field */}
                  <div>
                    <label className="block text-gray-600 font-medium mb-1">Mobile Number</label>
                    <input
                      type="text"
                      name="phone_number"
                      value={formData.phone_number}
                      onChange={handleInputChange}
                      placeholder="Add Number"
                      readOnly={!isEditMode}
                      className="w-full border border-gray-300 rounded-full px-4 py-2 focus:outline-none focus:ring-2 focus:ring-[#6966FF]"
                    />
                  </div>

                  {/* Location Field */}
                  <div>
                    <label className="block text-gray-600 font-medium mb-1">Location</label>
                    <input
                      type="text"
                      name="location"
                      value={formData.location}
                      onChange={handleInputChange}
                      placeholder="Location"
                      readOnly={!isEditMode}
                      className="w-full border border-gray-300 rounded-full px-4 py-2 focus:outline-none focus:ring-2 focus:ring-[#6966FF]"
                    />
                  </div>

                </div>

                {/* Right Section (Roles) */}
                <div className="lg:w-1/2 mt-10 ">
                  <h3 className="text-xl font-medium text-[#39407D] mb-4">
                    User Roles
                  </h3>
                  <div className="space-y-3">
                    {rolesList.map((role) => (
                      <div
                        key={role}
                        className="flex items-center justify-between bg-gray-100 rounded-lg p-3 shadow-sm"
                        onClick={() => isEditMode && toggleRole(role)}
                        style={{ cursor: isEditMode ? "pointer" : "default" }}
                      >
                        <span className="text-gray-700">{role}</span>
                        <div
                          className={`w-6 h-6 flex items-center justify-center rounded-full ${
                            formData.roles.includes(role)
                              ? "bg-green-500"
                              : "bg-gray-300"
                          }`}
                        >
                          {formData.roles.includes(role) && (
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

              {/* Buttons */}
              <div className="mt-6 flex justify-center gap-4">
                {isEditMode && (
                  <button
                    type="submit"
                    className="bg-[#6966FF] text-white px-6 py-2 rounded-full shadow-lg"
                  >
                    Save
                  </button>
                )}
                <button
                  type="button"
                  onClick={() => setModalVisible(false)}
                  className="bg-gray-300 text-gray-800 px-6 py-2 rounded-full shadow"
                >
                  {isEditMode ? "Cancel" : "Close"}
                </button>
              </div>
            </form>
          </div>
        )}
      </div>
    </>
  );
};

export default ManageUser;
