<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>✨ Register</title>
    <link rel="stylesheet" href="<?=base_url();?>/public/css/style.css">
</head>
<body>
    <!-- Main Content -->
    <div class="main-content">
        <div class="center-box">
            <h1>✨ Register</h1>

            <p style="font-size:12px; color:#666;">
                Password must be at least 8 characters, include one uppercase, one lowercase, one number, and a special character.
            </p>
            <?php flash_alert(); ?>

            <form id="regForm" method="post" action="<?=site_url('auth/register');?>">

                <!-- CSRF -->
                <?php csrf_field(); ?>

                <!-- Username -->
                <div class="form-group">
                    <label for="username">Username</label>
                    <input id="username" type="text" class="form-control" name="username" required>
                </div>

                <!-- Email -->
                <div class="form-group">
                    <label for="email">Email Address</label>
                    <input id="email" type="email" class="form-control" name="email" required>
                </div>

                <!-- Password -->
                <div class="form-group">
                    <label for="password">Password</label>
                    <input id="password" type="password" class="form-control" name="password" required minlength="8">
                </div>

                <!-- Confirm Password -->
                <div class="form-group">
                    <label for="password_confirmation">Confirm Password</label>
                    <input id="password_confirmation" type="password" class="form-control" name="password_confirmation" required minlength="8">
                </div>

                <div class="form-group">
                    <label for="role">Role</label>
                    <select id="role" name="role" type="role" class="form-control" required>
                        <option value="fairy">Fairy</option>
                        <option value="admin">Admin</option>
                    </select>
                </div>

                <button type="submit" class="submit-btn">✨ Register</button>
            </form>

            <!-- Back -->
            <p style="margin-top: 15px;">
                Already have an account? <a href="<?=site_url('auth/login');?>">✨ Login here</a>
            </p>
        </div>
    </div>

    <!-- Validation -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.1/jquery.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/jquery-validation@1.19.5/dist/jquery.validate.js"></script>
    <script>
        $(function() {
            $("#regForm").validate({
                rules: {
                    username: { required: true, minlength: 5, maxlength: 20 },
                    email: { required: true, email: true },
                    password: { required: true, minlength: 8 },
                    password_confirmation: { required: true, minlength: 8, equalTo: "#password" },
                    role: { required: true}
                },
                messages: {
                    username: { required: "Please input your username." },
                    email: { required: "Please input your email address." },
                    password: { required: "Please input your password", minlength: "Password must be at least 8 characters." },
                    password_confirmation: { required: "Please confirm your password", minlength: "Password must be at least 8 characters.", equalTo: "Passwords do not match." },
                    role: { required: "Please select your role."}
                }
            });
        });
    </script>
</body>
</html>
