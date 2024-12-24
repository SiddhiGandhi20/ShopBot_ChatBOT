import React, { useState } from 'react';
import './App.css';

// Chatbot component
function App() {
  const [userInput, setUserInput] = useState(''); // State for user input
  const [messages, setMessages] = useState([]); // State for storing chat messages

  const handleInputChange = (event) => {
    setUserInput(event.target.value); // Update user input
  };

  const handleSendMessage = () => {
    if (userInput.trim() === '') return;

    // Add user input to message history
    setMessages((prevMessages) => [
      ...prevMessages,
      { sender: 'user', text: userInput },
    ]);

    // Send the user input to the backend (Python server)
    fetch('http://localhost:5000/chatbot', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: userInput }),
    })
      .then((response) => response.json())
      .then((data) => {
        // Add bot response to message history
        setMessages((prevMessages) => [
          ...prevMessages,
          { sender: 'bot', text: data.reply },
        ]);
      })
      .catch((error) => {
        console.error('Error:', error);
        setMessages((prevMessages) => [
          ...prevMessages,
          { sender: 'bot', text: "I'm sorry, there was an error." },
        ]);
      });

    setUserInput(''); // Clear the input field
  };

  return (
    <div className="App">
      <div className="chat-window">
        <div className="chat-history">
          {messages.map((message, index) => (
            <div
              key={index}
              className={message.sender === 'user' ? 'user-message' : 'bot-message'}
            >
              <p>{message.text}</p>
            </div>
          ))}
        </div>

        <div className="input-container">
          <input
            type="text"
            value={userInput}
            onChange={handleInputChange}
            placeholder="Type your message..."
          />
          <button onClick={handleSendMessage}>Send</button>
        </div>
      </div>
    </div>
  );
}

export default App;
