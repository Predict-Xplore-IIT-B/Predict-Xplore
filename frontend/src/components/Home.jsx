import React from 'react';
import '../styles/Home.css';
import Navbar from './Navbar.jsx';
const Home = () => {
  const models = [
    { title: 'Model 1', date: '17 JUL 2022', description: 'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Aenean eu fermentum augue, sit amet convallis augue. Integer eu iaculis sem, sed euismod eros. Nulla facilisi. Proin luctus odio nunc, sed laoreet est bibendum vitae. Sed a eleifend ex. Integer varius rhoncus euismod. Aliquam ac ultricies turpis, vitae eleifend ligula. Aliquam faucibus erat ut tincidunt cursus. Cras et ullamcorper velit.' },
    { title: 'Model 2', date: '17 JUL 2022', description: 'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Aenean eu fermentum augue, sit amet convallis augue. Integer eu iaculis sem, sed euismod eros. Nulla facilisi. Proin luctus odio nunc, sed laoreet est bibendum vitae. Sed a eleifend ex. Integer varius rhoncus euismod. Aliquam ac ultricies turpis, vitae eleifend ligula. Aliquam faucibus erat ut tincidunt cursus. Cras et ullamcorper velit.' },
    { title: 'Model 3', date: '17 JUL 2022', description: 'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Aenean eu fermentum augue, sit amet convallis augue. Integer eu iaculis sem, sed euismod eros. Nulla facilisi. Proin luctus odio nunc, sed laoreet est bibendum vitae. Sed a eleifend ex. Integer varius rhoncus euismod. Aliquam ac ultricies turpis, vitae eleifend ligula. Aliquam faucibus erat ut tincidunt cursus. Cras et ullamcorper velit.' },
    { title: 'Model 4', date: '17 JUL 2022', description: 'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Aenean eu fermentum augue, sit amet convallis augue. Integer eu iaculis sem, sed euismod eros. Nulla facilisi. Proin luctus odio nunc, sed laoreet est bibendum vitae. Sed a eleifend ex. Integer varius rhoncus euismod. Aliquam ac ultricies turpis, vitae eleifend ligula. Aliquam faucibus erat ut tincidunt cursus. Cras et ullamcorper velit.' }
  ];

  return (
    <>
      <div className="home-container">
        <div className="white-box-1"></div>
        
        <h2 className="title">OUR MODELS</h2>
        <div className="models-container">
          {models.map((model, index) => (
            <div key={index} className="model-card">
              <h3 className="model-title">♦ {model.title}</h3>
              <div className="model-line"></div>
              <p className="model-date">{model.date}</p>
              <p className="model-description">{model.description}</p>
            </div>
          ))}
        </div>
      </div>
    </>
  );
};

export default Home;