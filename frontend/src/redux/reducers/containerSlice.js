import { createSlice } from "@reduxjs/toolkit";

const containerSlice = createSlice({
  name: "container",
  initialState: {
    containers: [],
    selected: [],
  },
  reducers: {
    setContainers: (state, action) => {
      state.containers = action.payload;
    },
    toggleContainerSelection: (state, action) => {
      const id = action.payload;
      if (state.selected.includes(id)) {
        state.selected = state.selected.filter((c) => c !== id);
      } else {
        state.selected.push(id);
      }
    },
  },
});

export const { setContainers, toggleContainerSelection } = containerSlice.actions;
export default containerSlice.reducer;
