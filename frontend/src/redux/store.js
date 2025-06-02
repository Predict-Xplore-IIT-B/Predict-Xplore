import { configureStore } from "@reduxjs/toolkit";
import userReducer from "./reducers/userSlice";
import modelReducer from "./reducers/modelSlice";

export const store = configureStore({
  reducer: {
    user: userReducer, 
    models: modelReducer,
  },
});

export default store;