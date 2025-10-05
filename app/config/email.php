<?php
defined('PREVENT_DIRECT_ACCESS') OR exit('No direct script access allowed');

$config = [
    'protocol'  => 'smtp',
    'smtp_host' => 'smtp.gmail.com',
    'smtp_user' => getenv('SMTP_USER'),
    'smtp_pass' => getenv('SMTP_PASS'),
    'smtp_port' => 587,
    'mailtype'  => 'html',
    'charset'   => 'utf-8',
    'newline'   => "\r\n",
    'smtp_crypto' => 'tls',
    'smtp_debug'  => 2, // 👈 depende kung sinusuportahan ng LavaLust
];

