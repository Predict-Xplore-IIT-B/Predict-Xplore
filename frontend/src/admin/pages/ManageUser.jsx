import React, { useEffect, useState } from "react";
import { MdEdit, MdDelete } from "react-icons/md";
import AdminNavbar from "../../components/AdminNavbar";
import userProfile from "../../assets/userProfile.png";
import { useSelector } from "react-redux";

const API_BASE = "http://localhost:8000/auth/admin/users";

const rolesList = [
  "Cse_dept",
  "AIML_dept",
  "aerospace_dept",
  "cyber_dept",
  "mech_dept",
];

const ManageUser = () => {
  const token = useSelector((state) => state.user.users?.[0]?.token);

  if (!token) {
    return (
      <div className="text-center text-xl mt-20 text-black">
        Please log in to access user management.
      </div>
    );
  }

  const AUTH_HEADERS = {
    "Content-Type": "application/json",
    Authorization: `Token ${token}`,
  };
  const [users, setUsers] = useState([]);
  const [modalVisible, setModalVisible] = useState(false);
  const [isEditMode, setIsEditMode] = useState(false);
  const [formData, setFormData] = useState({
    id: null,
    name: "",
    username: "",
    phone_number: "",
    email: "",
    user_roles: [],
    location: "",
  });

  // Centralized fetch function
  const fetchUsers = async () => {
    try {
      const res = await fetch(API_BASE, { headers: AUTH_HEADERS });
      if (!res.ok) throw new Error("Failed to fetch users");
      const data = await res.json();
      setUsers(data);
    } catch (err) {
      console.error("Fetch users failed:", err);
    }
  };

  useEffect(() => {
    fetchUsers();
  }, []);

  const openAddModal = () => {
    setFormData({
      id: null,
      name: "",
      username: "",
      phone_number: "",
      email: "",
      user_roles: [],
      location: "",
    });
    setIsEditMode(true);
    setModalVisible(true);
  };

  const openEditModal = (user) => {
    setFormData({
      id: user.id,
      name: user.username,
      username: user.username,
      phone_number: user.phone_number || "",
      email: user.email || "",
      user_roles: user.user_roles || [],
      location: "",
    });
    setIsEditMode(true);
    setModalVisible(true);
  };

  const openViewModal = (user) => {
    setFormData({
      id: user.id,
      name: user.username,
      username: user.username,
      phone_number: user.phone_number || "",
      email: user.email || "",
      user_roles: user.user_roles || [],
      location: "",
    });
    setIsEditMode(false);
    setModalVisible(true);
  };

  const closeModal = () => setModalVisible(false);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const handleRoleToggle = (role) => {
    setFormData((prev) => ({
      ...prev,
      user_roles: prev.user_roles.includes(role)
        ? prev.user_roles.filter((r) => r !== role)
        : [...prev.user_roles, role],
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const isCreating = !formData.id;
    const payload = {
      ...(formData.id && { id: formData.id }),
      username: formData.username,
      email: formData.email,
      phone_number: formData.phone_number,
      user_roles: formData.user_roles,
      is_active: true,
      ...(isCreating && { password: "12345" }),
    };

    try {
      const method = isCreating ? "POST" : "PUT";
      const response = await fetch(API_BASE, {
        method,
        headers: AUTH_HEADERS,
        body: JSON.stringify(payload),
      });

      if (response.ok) {
        await fetchUsers(); // Refresh the list from backend
        closeModal();
      } else {
        const errData = await response.json();
        console.error("Save error:", errData);
        alert("Failed to save user. See console.");
      }
    } catch (error) {
      console.error("Submit failed:", error);
      alert("Request failed. Check console.");
    }
  };

  const handleDelete = async (id) => {
    if (!window.confirm("Are you sure you want to delete this user?")) return;
    try {
      const response = await fetch(API_BASE, {
        method: "DELETE",
        headers: AUTH_HEADERS,
        body: JSON.stringify({ id }),
      });
      if (response.ok) {
        await fetchUsers(); // Refetch the user list
      } else {
        const text = await response.text();
        console.error("Delete failed:", text);
      }
    } catch (err) {
      console.error("Delete error:", err);
    }
  };

  return (
    <>
      <AdminNavbar />
      <div className="bg-[#EAECFF] min-h-screen p-6">
        <div className="container mx-auto">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-3xl font-semibold text-[#39407D]">
              Manage Users
            </h2>
            <button
              className="bg-[#6966FF] text-white px-4 py-2 rounded-full shadow-lg"
              onClick={openAddModal}
            >
              ➕ Add User
            </button>
          </div>

          <div className="grid grid-cols-5 font-semibold text-[#39407D] mb-2 ml-7 gap-14">
            <div>Username</div>
            <div>Email</div>
            <div>Phone</div>
            <div>Roles</div>
            <div>Actions</div>
          </div>

          {users.map((user) => (
            <div
              key={user.id}
              className="grid grid-cols-5 bg-white p-4 border border-[#9f9df1] rounded-xl mb-2 shadow hover:bg-gray-100 cursor-pointer  gap-16"
            >
              {/* Row content (make these divs clickable to view details) */}
              <div onClick={() => openViewModal(user)}>{user.username}</div>
              <div onClick={() => openViewModal(user)}>{user.email}</div>
              <div onClick={() => openViewModal(user)}>{user.phone_number}</div>
              <div onClick={() => openViewModal(user)}>
                {(user.user_roles || []).join(", ")}
              </div>

              {/* Action buttons */}
              <div className="flex space-x-3 items-center ml-4 ">
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    openEditModal(user);
                  }}
                >
                  <MdEdit className="text-blue-500 text-xl " />
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    handleDelete(user.id);
                  }}
                >
                  <MdDelete className="text-red-500 text-xl" />
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

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
              <div className="lg:w-1/2 space-y-4">
                <div className="flex items-center gap-4">
                  <img
                    src={userProfile}
                    alt="Profile"
                    className="w-16 h-16 rounded-full shadow-md"
                  />
                  <div>
                    <h2 className="text-lg font-semibold text-[#39407D]">
                      {formData.name || "Name"}
                    </h2>
                    <p className="text-gray-600">
                      {formData.email || "email@example.com"}
                    </p>
                  </div>
                </div>

                {["name", "username", "email", "phone_number", "location"].map(
                  (field) => (
                    <div key={field}>
                      <label className="block text-gray-600 font-medium mb-1 capitalize">
                        {field.replace("_", " ")}
                      </label>
                      <input
                        type={field === "email" ? "email" : "text"}
                        name={field}
                        value={formData[field]}
                        onChange={handleInputChange}
                        placeholder={field}
                        readOnly={!isEditMode}
                        className="w-full border border-gray-300 rounded-full px-4 py-2 focus:outline-none focus:ring-2 focus:ring-[#6966FF]"
                      />
                    </div>
                  )
                )}
              </div>

              <div className="lg:w-1/2 mt-10">
                <h3 className="text-xl font-medium text-[#39407D] mb-4">
                  User Roles
                </h3>
                <div className="space-y-3">
                  {rolesList.map((role) => (
                    <div
                      key={role}
                      className="flex items-center justify-between bg-gray-100 rounded-lg p-3 shadow-sm"
                      onClick={() => isEditMode && handleRoleToggle(role)}
                      style={{ cursor: isEditMode ? "pointer" : "default" }}
                    >
                      <span className="text-gray-700">{role}</span>
                      <div
                        className={`w-6 h-6 flex items-center justify-center rounded-full ${
                          formData.user_roles.includes(role)
                            ? "bg-green-500"
                            : "bg-gray-300"
                        }`}
                      >
                        {formData.user_roles.includes(role) && (
                          <svg
                            className="w-4 h-4 text-white"
                            fill="none"
                            stroke="white"
                            strokeWidth="3"
                            viewBox="0 0 24 24"
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

            <div className="mt-6 flex justify-center gap-4">
              {isEditMode && (
                <button
                  type="submit"
                  className="bg-[#6966FF] text-white px-6 py-2 rounded-full shadow-lg"
                >
                  {formData.id ? "Update" : "Create"}
                </button>
              )}
              <button
                type="button"
                onClick={closeModal}
                className="bg-gray-300 text-gray-800 px-6 py-2 rounded-full shadow"
              >
                {isEditMode ? "Cancel" : "Close"}
              </button>
            </div>
          </form>
        </div>
      )}
    </>
  );
};

export default ManageUser;
