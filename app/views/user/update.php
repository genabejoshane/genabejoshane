<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>User/Update</title>
  <style>
    body {
      font-family: 'Poppins', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      background: #e6f4ea; /* light green background */
      margin: 0;
      padding: 0;
      display: flex;
      justify-content: center;
      align-items: center;
      height: 100vh;
    }

    .form-container {
      background: #e6f4ea;
      padding: 40px;
      border-radius: 25px;
      width: 380px;
      box-shadow: 8px 8px 20px #b8c9be, -8px -8px 20px #ffffff;
      animation: fadeIn 0.6s ease-in-out;
    }

    .form-container h1 {
      text-align: center;
      margin-bottom: 25px;
      font-size: 24px;
      font-weight: 600;
      color: #1e3d2f; /* dark green heading */
    }

    label {
      font-weight: 500;
      display: block;
      margin-top: 15px;
      margin-bottom: 6px;
      color: #2e5c46; /* darker green labels */
    }

    input[type="text"], input[type="email"] {
      width: 100%;
      padding: 12px;
      border: none;
      border-radius: 15px;
      outline: none;
      background: #e6f4ea;
      box-shadow: inset 4px 4px 8px #b8c9be, inset -4px -4px 8px #ffffff;
      font-size: 15px;
      transition: all 0.3s ease;
      color: #1e3d2f; /* dark green text */
    }

    input[type="text"]:focus, input[type="email"]:focus {
      box-shadow: inset 2px 2px 6px #b8c9be, inset -2px -2px 6px #ffffff;
    }

    input[type="submit"] {
      margin-top: 30px;
      width: 100%;
      padding: 14px;
      background: #1e3d2f; /* dark green button */
      color: #e6f4ea; /* light text */
      font-size: 16px;
      font-weight: 600;
      border: none;
      border-radius: 30px;
      cursor: pointer;
      box-shadow: 6px 6px 12px #b8c9be, -6px -6px 12px #ffffff;
      transition: all 0.3s ease;
    }

    input[type="submit"]:hover {
      transform: translateY(-2px);
      background: #2e5c46; /* slightly lighter dark green */
      box-shadow: 4px 4px 10px rgba(30, 61, 47, 0.4), -4px -4px 10px #ffffff;
    }

    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(-20px); }
      to { opacity: 1; transform: translateY(0); }
    }
  </style>
</head>
<body>
  <div class="form-container">
    <h1>Update User</h1>
    <form method="post" action="<?= site_url('user/update/'.$user['id']) ?>">
      <label for="username">Username:</label>
      <input type="text" name="username" id="username" value="<?= html_escape($user['username']) ?>" required>

      <label for="email">Email:</label>
      <input type="email" name="email" id="email" value="<?= html_escape($user['email']) ?>" required>

      <input type="submit" value="Update">
    </form>
  </div>
</body>
</html>
