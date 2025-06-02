import { createSlice } from "@reduxjs/toolkit";

const initialState = {
    users: []
};

export const userSlice = createSlice({
    name: "user",
    initialState,
    reducers: {
        addUser: (state, action) => {
            const newUser = {
                username: action.payload.username,
                email: action.payload.email,
                phone_number: action.payload.phone_number,
                role: action.payload.role,
                password: action.payload.password,
                token: "", // Initialize token as empty
                isActive: false // Default inactive after registration
            };
            state.users.push(newUser);
        },

        removeUser: (state, action) => {
            state.users = state.users.filter(user => user.username !== action.payload);
        },

        updateUserStatus: (state, action) => {
            const { username, token, isActive } = action.payload;
            const user = state.users.find(user => user.username === username);
            if (user) {
                user.token = token;
                user.isActive = isActive;
            }
        },
        addLogedUser: (state, action) => {
            const {username, phone_number, role, roles, email, password, token, isActive } =  action.payload;
            state.users = []
            const newUser = {
                username:username,
                phone_number: phone_number,
                role: role,
                roles: roles,
                email: email,
                password: password,
                token: token, // Initialize token as empty
                isActive: isActive // Default inactive after registration
            }
            state.users.push(newUser);
        }
    }
});

export const { addUser, addLogedUser,removeUser, updateUserStatus } = userSlice.actions;

export default userSlice.reducer;
