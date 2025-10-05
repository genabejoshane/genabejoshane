<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login Magic User</title>
    <link rel="icon" type="image/png" href="<?=base_url();?>public/img/favicon.ico"/>
    <link rel="stylesheet" href="<?=base_url();?>/public/css/style.css">

    <!-- Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Nosifer&display=swap" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Creepster&display=swap" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Cinzel+Decorative:wght@700&display=swap" rel="stylesheet">
</head>
<body>
  
    <!-- Main Content -->
    <div class="main-content">
        <div class="center-box">
            <h1>✨ Login Magic User</h1>

            <?php flash_alert(); ?>
            <form id="logForm" method="POST" action="<?=site_url('auth/login');?>"> 
                <?php csrf_field(); ?>

                <!-- Email -->
                <div class="form-group">
                    <label for="email">Email Address</label>
                    <input id="email" type="email" name="email" placeholder="you@example.com" required>
                </div>

                <!-- Password -->
                <div class="form-group">
                    <label for="password">Password</label>
                    <input id="password" type="password" name="password" placeholder="Enter password" minlength="8" required>
                </div>

                <!-- Submit -->
                <button type="submit" class="submit-btn">✨ Log In</button>
            </form>

            <!-- Forgot Password -->
            <div class="form-group" style="margin-top: 10px; text-align: center;">
                <a href="<?=site_url('auth/password-reset');?>">Forgot Your Password?</a>
            </div>

            <!-- Navigation Links -->
            <p style="margin-top: 15px;">
                Don’t have an account? <a href="<?=site_url('auth/register');?>">✨ Register here</a>
            </p>

            <!-- Back Button -->
            <a href="<?=site_url('/');?>" class="back-btn">← Go Back</a>
        </div>
    </div>

    <!-- Scripts -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.1/jquery.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/jquery-validation@1.19.5/dist/jquery.validate.js"></script>
    <script>
        $(function() {
            var logForm = $("#logForm");
            if(logForm.length) {
                logForm.validate({
                    rules: {
                        email: { required: true },
                        password: { required: true }
                    },
                    messages: {
                        email: { required: "Please input your email address." },
                        password: { required: "Please input your password." }
                    }
                });
            }
        });
    </script>
</body>
</html>
