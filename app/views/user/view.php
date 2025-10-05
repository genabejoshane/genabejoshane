<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>User View</title>
  <style>
    body {
      font-family: 'Poppins', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      background: #e6f4ea; /* light green */
      margin: 0;
      padding: 20px;
      color: #1e3d2f; /* dark green text */
    }

    h1 {
      text-align: center;
      margin-bottom: 30px;
      font-size: 28px;
      font-weight: 600;
      color: #1e3d2f;
    }

    .create-btn {
      display: inline-block;
      margin-bottom: 20px;
      background: linear-gradient(135deg, #1e3d2f, #2e5c46); /* green gradient */
      color: white;
      padding: 12px 24px;
      border-radius: 30px;
      font-weight: bold;
      font-size: 15px;
      text-decoration: none;
      box-shadow: 0 4px 12px rgba(30, 61, 47, 0.3);
      transition: all 0.3s ease;
    }

    .create-btn:hover {
      transform: translateY(-2px);
      background: #2e5c46;
      box-shadow: 0 6px 16px rgba(30, 61, 47, 0.45);
    }

    table {
      width: 85%;
      margin: 0 auto;
      border-collapse: collapse;
      background: #ffffff;
      border-radius: 15px;
      overflow: hidden;
      box-shadow: 0 8px 20px rgba(0,0,0,0.1);
      animation: fadeIn 0.6s ease-in-out;
    }

    th, td {
      padding: 16px 20px;
      text-align: center;
      font-size: 15px;
    }

    th {
      background: #1e3d2f; /* dark green */
      color: #fff;
      text-transform: uppercase;
      font-size: 14px;
      letter-spacing: 0.5px;
    }

    tr:nth-child(even) {
      background: #f0f9f3; /* very light green */
    }

    tr:nth-child(odd) {
      background: #ffffff;
    }

    tr:hover {
      background: #d9f2e1; /* hover green */
      transition: 0.3s;
    }

    .action-btn {
      display: inline-block;
      padding: 6px 14px;
      border-radius: 20px;
      font-size: 13px;
      font-weight: 600;
      text-decoration: none;
      transition: 0.3s;
    }

    .edit-btn {
      background: #2e7d32; /* leafy green */
      color: #fff;
      margin-right: 8px;
      box-shadow: 0 2px 6px rgba(46, 125, 50, 0.3);
    }

    .edit-btn:hover {
      background: #256428;
    }

    .delete-btn {
      background: #e53935; /* red for delete */
      color: #fff;
      box-shadow: 0 2px 6px rgba(229, 57, 53, 0.3);
    }

    .delete-btn:hover {
      background: #b71c1c;
    }

    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(-10px); }
      to { opacity: 1; transform: translateY(0); }
    }
  </style>
</head>
<body>
  <h1> Welcome to Profile View</h1>

  <div style="width: 85%; margin: 0 auto 25px auto; text-align: right;">
    <a href="<?= site_url('user/create'); ?>" class="create-btn">+ Create New User</a>
  </div>

  <table>
    <tr>
      <th>ID</th>
      <th>Username</th>
      <th>Email</th>
      <th>Action</th>
    </tr>

    <?php foreach ($users as $user): ?>
      <tr>
        <td><?= $user['id']; ?></td>
        <td><?= $user['username']; ?></td>
        <td><?= $user['email']; ?></td>
        <td>
          <a href="<?= site_url('user/update/'.$user['id']); ?>" class="action-btn edit-btn">Edit</a>
          <a href="<?= site_url('user/delete/'.$user['id']); ?>" class="action-btn delete-btn"
             onclick="return confirm('Are you sure you want to delete this user?');">Delete</a>
        </td>
      </tr>
    <?php endforeach; ?>
  </table>
</body>
</html>
