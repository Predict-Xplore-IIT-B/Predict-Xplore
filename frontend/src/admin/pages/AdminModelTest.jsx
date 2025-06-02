import React from 'react'
import ModelTest from '../../components/ModelTest'
import Navbar from '../../components/Navbar'
import AdminNavbar from '../../components/AdminNavbar'

function AdminModelTest() {
  return (
    <div className="bg-[#EAECFF]">
      <AdminNavbar/>
      {/* search bar */}
      <div className="">
        <input type="search" className="relative block w-[35%] min-w-0 rounded-full border border-solid border-neutral-300 bg-clip-padding px-3 py-[0.25rem] text-base leading-[1.6]  outline-none transition duration-200 ease-in-out focus:z-[3] focus:border-primary focus:text-neutral-700 focus:shadow-[inset_0_0_0_1px_rgb(59,113,202)] focus:outline-none  bg-white h-[5vh] m-8 ml-[3.25rem]"
          id="exampleSearch" placeholder="Enter name of the model" />
      </div>

      <div>

        <ModelTest />
      </div>
    </div>
  )
}

export default AdminModelTest