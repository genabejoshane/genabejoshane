<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>User/Create</title>
  <style>
    body {
      font-family: 'Poppins', 'Segoe UI', sans-serif;
      background: linear-gradient(135deg, #1e3c72, #2a5298);
      margin: 0;
      padding: 0;
      display: flex;
      justify-content: center;
      align-items: center;
      height: 100vh;
    }

    .form-container {
      background: rgba(255, 255, 255, 0.15);
      backdrop-filter: blur(15px);
      -webkit-backdrop-filter: blur(15px);
      border-radius: 20px;
      padding: 35px 45px;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
      width: 380px;
      color: #fff;
      animation: fadeIn 0.7s ease-in-out;
    }

    .form-container h1 {
      text-align: center;
      margin-bottom: 25px;
      font-size: 24px;
      font-weight: 600;
      color: #f0f0f0;
    }

    label {
      display: block;
      margin-top: 15px;
      margin-bottom: 6px;
      font-size: 14px;
      color: #dcdcdc;
      font-weight: 500;
    }

    input[type="text"], input[type="email"] {
      width: 100%;
      padding: 12px;
      border: none;
      border-radius: 12px;
      outline: none;
      background: rgba(255, 255, 255, 0.2);
      color: #fff;
      font-size: 15px;
      transition: 0.3s;
    }

    input[type="text"]::placeholder,
    input[type="email"]::placeholder {
      color: rgba(255, 255, 255, 0.6);
    }

    input[type="text"]:focus, input[type="email"]:focus {
      background: rgba(255, 255, 255, 0.3);
      box-shadow: 0 0 0 2px #00c6ff;
    }

    input[type="submit"] {
      margin-top: 25px;
      width: 100%;
      padding: 14px;
      background: linear-gradient(135deg, #00c6ff, #0072ff);
      color: white;
      font-size: 16px;
      font-weight: bold;
      border: none;
      border-radius: 30px;
      cursor: pointer;
      transition: 0.3s;
    }

    input[type="submit"]:hover {
      background: linear-gradient(135deg, #0072ff, #00c6ff);
      transform: translateY(-2px);
      box-shadow: 0 6px 16px rgba(0, 198, 255, 0.5);
    }

    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(-20px); }
      to { opacity: 1; transform: translateY(0); }
    }
  </style>
</head>
<body>
  <div class="form-container">
    <h1>Create New User</h1>
    <form method="post" action="">
      <label for="username">Username</label>
      <input type="text" name="username" id="username" placeholder="Enter username" required>

      <label for="email">Email</label>
      <input type="email" name="email" id="email" placeholder="Enter email" required>

      <input type="submit" value="Create New User">
    </form>
  </div>
</body>
</html>
